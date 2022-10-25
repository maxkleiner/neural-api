{
unit uvisualgan       script-  work still on progress!
Copyright (C) 2018 Joao Paulo Schwarz Schuler

BuildTrainingPairs(); crash now fixed

This program is free software; you can redistribute it and/or modify   
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

https://forum.lazarus.freepascal.org/index.php?topic=39049.0
https://sourceforge.net/p/cai/svncode/HEAD/tree/trunk/lazarus/neural/
https://sourceforge.net/p/cai/svncode/HEAD/tree/trunk/lazarus/examples/VisualGAN/uvisualgan.pas
}

unit uvisualgan_mX41_2;

//{$mode objfpc}{$H+}

interface

//uses
  {$ifdef unix}
  cmem, // the c memory manager is on some systems much faster for multi-threading
  {$endif}
  {Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls,
  ExtCtrls, Menus, neuralnetwork, neuralvolumev, neuraldatasets,
  neuralvolume, MTProcs, math, neuralfit }

  {$ifdef OpenCL}
  , neuralopencl
  {$endif}
  //;

var csLearningRates: array[0..2] of TNeuralFloat; // = (1, 0.1, 0.01);

type

  { TFormVisualLearning }
  TFormVisualLearning = {class(}TForm;
  var
    ButLearn: TButton;
    ChkRunOnGPU: TCheckBox;
    ChkBigNetwork: TCheckBox;
    ComboLearningRate: TComboBox;
    ComboComplexity: TComboBox;
    GrBoxNeurons: TGroupBox;
    ImgSample: TImage;
    LabClassRate: TLabel;
    LabComplexity: TLabel;
    LabLearningRate: TLabel;
    RadLAB: TRadioButton;
    RadRGB: TRadioButton;
    procedure TFormVisualLearningButLearnClick(Sender: TObject);
    procedure TFormVisualLearningFormClose(Sender:TObject; var CloseAction:TCloseAction);
    procedure TFormVisualLearningFormCreate(Sender: TObject);
    procedure TFormVisualLearningFormDestroy(Sender: TObject);
  //private
  var
    { private declarations }
    FRunning: boolean;
    FDisplay: TNNetVolume;
    ImgTrainingVolumes, ImgTestVolumes, ImgValidationVolumes: TNNetVolumeList;
    FRealPairs, FGeneratedPairs: TNNetVolumePairList;
    FImageCnt: integer;
    iEpochCount, iEpochCountAfterLoading: integer;
    FGenerative: THistoricalNets;
    FGeneratives: TNNetDataParallelism;
    FDiscriminator, FDiscriminatorClone: TNNet;
    aImage: array of TImage;
    aLabelX, aLabelY: array of TLabel;
    FBaseName: string;
    FColorEncoding: byte;
    FRandomSizeX, FRandomSizeY, FRandomDepth: integer;
    FLearningRateProportion: TNeuralFloat;
    {$ifdef OpenCL}
    FEasyOpenCL: TEasyOpenCL;
    FHasOpenCL: boolean;
    {$endif}

    FCritSec: TRTLCriticalSection;
    FFit: TNeuralDataLoadingFit;
    function GetDiscriminatorTrainingPair(Idx: integer; ThreadId: integer): TNNetVolumePair;
    procedure GetDiscriminatorTrainingProc(Idx: integer; ThreadId: integer; 
              pInput, pOutput: TNNetVolume);
    procedure DiscriminatorOnAfterEpoch(Sender: TObject);
    procedure DiscriminatorOnAfterStep(Sender: TObject);
    procedure DiscriminatorAugmentation(pInput: TNNetVolume; ThreadId: integer);
    procedure TFormVisualLearningLearn(Sender: TObject);
    procedure TFormVisualLearningSaveScreenshot(filename: string);
    procedure BuildTrainingPairs();
    procedure DisplayInputImage(ImgInput: TNNetVolume; color_encoding: integer);
    procedure DiscriminatorOnStart(Sender: TObject);
    procedure TFormVisualLearningSendStop;
  //public
    procedure TFormVisualLearningProcessMessages();
 // end;

var
  FormVisualLearning: TFormVisualLearning;

implementation
//{$R *.lfm}

//uses strutils, LCLIntf, LCLType, neuraldatasetsv;


{ TFormVisualLearning }

procedure TFormVisualLearningButLearnClick(Sender: TObject);
begin
  if not CheckCIFARFile() then exit;
  writeln('CheckCIFARFile: '+botostr(CheckCIFARFile));

  if (FRunning) then begin
    TFormVisualLearningSendStop;
  end else begin
    FRunning := true;
    ButLearn.Caption := 'Stop';
    ChkBigNetwork.Enabled := false;
    TFormVisualLearningLearn(Sender);
    ChkBigNetwork.Enabled := true;
    ButLearn.Caption := 'Restart';
    FRunning := false;
  end;
end;

procedure TFormVisualLearningFormClose(Sender: TObject;
  var CloseAction: TCloseAction);
begin
  TFormVisualLearningSendStop;
  writeln('form close_');
  closeaction:= caFree;  //triggers form destroy
end;

procedure TFormVisualLearningFormCreate(Sender: TObject);
begin
  FRunning := false;
  FFit := TNeuralDataLoadingFit.Create();
  InitCriticalSection(FCritSec);
  ImgTrainingVolumes := nil;
  ImgTestVolumes := nil;
  ImgValidationVolumes := nil;
  FGeneratives := nil;
  FRealPairs := TNNetVolumePairList.Create(true);
  FGeneratedPairs := TNNetVolumePairList.Create(true);
  FDisplay := TNNetVolume.Create();
  FImageCnt := 0;
  CreateAscentImages
  (
    GrBoxNeurons,
    aImage, aLabelX, aLabelY,
    {ImageCount=}128,
    {InputSize=}32, {FFilterSize=}64, {ImagesPerRow=}16
  );
  {$ifdef OpenCL}
  FEasyOpenCL := TEasyOpenCL.Create();
  {$else}
    ChkRunOnGPU.Visible := false;
  {$endif}
end;

procedure TFormVisualLearningFormDestroy(Sender: TObject);
begin
  TFormVisualLearningSendStop;
  while FFit.Running do Application.ProcessMessages;
  while FRunning do Application.ProcessMessages;
  if Assigned(ImgValidationVolumes) then ImgValidationVolumes.Free;
  if Assigned(ImgTestVolumes) then ImgTestVolumes.Free;
  if Assigned(ImgTrainingVolumes) then ImgTrainingVolumes.Free;
  FreeNeuronImages(aImage, aLabelX, aLabelY);
  DoneCriticalSection(FCritSec);
  FRealPairs.Free;
  FGeneratedPairs.Free;
  FDisplay.Free;
  {$ifdef OpenCL}FEasyOpenCL.Free;{$endif}
  FFit.Free;
  writeln('form destroy__ ');
end;

procedure DisplayInputImage(ImgInput: TNNetVolume; color_encoding: integer);
var
  pMin0, pMax0: TNeuralFloat;
  pMin1, pMax1: TNeuralFloat;
  pMin2, pMax2: TNeuralFloat;
begin
// RegisterMethod('Procedure ReSize10( Original : TVolume);');
  FDisplay.Resize10(ImgInput);            
  FDisplay.Copy76(ImgInput);

  if color_encoding = csEncodeLAB then begin
    FDisplay.GetMinMaxAtDepth(0, pMin0, pMax0);
    FDisplay.GetMinMaxAtDepth(1, pMin1, pMax1);
    FDisplay.GetMinMaxAtDepth(2, pMin2, pMax2);
    pMax0 := MaxF(Abs(pMin0), Abs(pMax0));
    pMax1 := MaxF(Abs(pMin1), Abs(pMax1));
    pMax2 := MaxF(Abs(pMin2), Abs(pMax2));

    if pMax0 > 2 then begin
      FDisplay.MulAtDepth27(0, 2/pMax0);
    end;

    if pMax1 > 2 then begin
      FDisplay.MulAtDepth27(1, 2/pMax1);
    end;

    if pMax2 > 2 then begin
      FDisplay.MulAtDepth27(2, 2/pMax2);
    end;
  end
  else if FDisplay.GetMaxAbs() > 2 then begin
    FDisplay.NormalizeMax(2);
  end;

  //Debug only: FDisplay.PrintDebugChannel();
  FDisplay.NeuronalInputToRgbImg(color_encoding);

  LoadVolumeIntoTImage(FDisplay, aImage[FImageCnt], color_encoding);
  aImage[FImageCnt].Width := 64;
  aImage[FImageCnt].Height := 64;
  TFormVisualLearningProcessMessages();
  FImageCnt := (FImageCnt + 1) mod Length(aImage);
end;

procedure DiscriminatorOnStart(Sender: TObject);
begin
  //check  true
  writeln('DiscriminatorOnStart(Sender: TObject);');
  //constructor Create(CloneNN: TNNet; pSize: integer; pFreeObjects: Boolean = True);
  FGeneratives:= TNNetDataParallelism.Create74(FGenerative, FFit.MaxThreadNum, true);
  {$ifdef OpenCL}
  if FHasOpenCL then begin
    FGeneratives.EnableOpenCL(FEasyOpenCL.PlatformIds[0], FEasyOpenCL.Devices[0]);
  end;
  {$endif}
  BuildTrainingPairs();
end;

procedure TFormVisualLearningSendStop;
begin
  WriteLn('Sending STOP request');
  FFit.ShouldQuit := true;
  //TFormVisualLearningFormDestroy(self);
end;

type  TNNetGet2VolumesProc2 = procedure(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume); // of object;


procedure TFormVisualLearningLearn( Sender: TObject);
var
  NeuronMultiplier: integer;
begin
  FRandomSizeX :=  4;
  FRandomSizeY :=  4;
  FRandomDepth := StrToInt(ComboComplexity.Text);
  csLearningRates[0]:= 1; csLearningRates[1]:= 0.1; csLearningRates[2]:= 0.01;
  {$ifdef OpenCL}
  FHasOpenCL := false;
  csLearningRates[0]:= 1; csLearningRates[1]:= 0.1; csLearningRates[2]:= 0.01;
  if ChkRunOnGPU.Checked then begin
    if FEasyOpenCL.GetPlatformCount() > 0 then begin
      FEasyOpenCL.SetCurrentPlatform(FEasyOpenCL.PlatformIds[0]);
      if FEasyOpenCL.GetDeviceCount() > 0 then begin
        FHasOpenCL := true;
      end;
    end;
  end;
  {$endif}
  if ChkBigNetwork.Checked
    then NeuronMultiplier := 2
    else NeuronMultiplier := 1;
  FBaseName := 'ART'+IntToStr(FRandomDepth)+'-'+IntToStr(NeuronMultiplier)+'-';
  if RadRGB.Checked then begin
    FColorEncoding := csEncodeRGB;
    //FBaseName += 'RGB-';
    FBaseName:= fbasename +'RGB-';
  end else begin
    FColorEncoding := csEncodeLAB;
    //FBaseName += 'LAB-';
    FBaseName := fbasename+ 'LAB-';
  end;
  FormVisualLearning.Height := GrBoxNeurons.Top + GrBoxNeurons.Height + 10;
  FormVisualLearning.Width  := GrBoxNeurons.Left + GrBoxNeurons.Width + 10;
  TFormVisualLearningProcessMessages();
  if not(Assigned(ImgValidationVolumes)) then begin
    CreateCifar10Volumes(ImgTrainingVolumes,ImgValidationVolumes, ImgTestVolumes,
                                               FColorEncoding);
  end;

  iEpochCount := 0;
  iEpochCountAfterLoading := 0;
  writeln('Creating Neural Networks...');
  FGenerative := THistoricalNets.Create();
  FDiscriminator := TNNet.Create();

  FLearningRateProportion:= csLearningRates[ComboLearningRate.ItemIndex];

  if Not(FileExists(FBaseName+'generative.nn')) then begin
    WriteLn('Creating generative.');
    FGenerative.AddLayer49([
      TNNetInput.Create4(FRandomSizeX, FRandomSizeY, FRandomDepth),
      TNNetConvolutionReLU.Create(128 * NeuronMultiplier,3,1,1,0), //4x4
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionReLU.Create(128 * NeuronMultiplier,3,1,1,0),
      TNNetMovingStdNormalization.Create(),
      //RegisterMethod('Constructor Create46( pPoolSize : integer; pSpacing : integer);');
      TNNetDeMaxPool.Create46(2,0),
      TNNetConvolutionReLU.Create(64 * NeuronMultiplier,5,2,1,0), //8x8
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionReLU.Create(64 * NeuronMultiplier,3,1,1,0),
      TNNetMovingStdNormalization.Create(),
      TNNetDeMaxPool.Create46(2,0),
      TNNetConvolutionReLU.Create(64 * NeuronMultiplier,5,2,1,0), //16x16
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionReLU.Create(32 * NeuronMultiplier,3,1,1,0),
      TNNetMovingStdNormalization.Create(),
      TNNetDeMaxPool.Create46(2,0),
      TNNetConvolutionReLU.Create(32 * NeuronMultiplier,5,2,1,0), //32x32
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionReLU.Create(32 * NeuronMultiplier,3,1,1,0),
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionLinear.Create(3,3,1,1,0),
      //constructor Create(LowLimit, HighLimit, Leakiness: integer); overload;
      TNNetReLUL.Create8(-40, +40) // Protection against overflow
    ]);
    //check index 0
    FGenerative.Layers[FGenerative.GetFirstImageNeuronalLayerIdx(1)].InitBasicPatterns();
  end else begin
    WriteLn('Loading generative.');
    FGenerative.LoadFromFile(FBaseName+'generative.nn');
    showmessagebig('generative.nn loaded ');
  end;
  FGenerative.DebugStructure();
  FGenerative.SetLearningRate(0.001,0.9);
  FGenerative.SetL2Decay(0.0);

  if Not(FileExists(FBaseName+'discriminator.nn')) then begin
    WriteLn('Creating discriminator.');
    (*
    FDiscriminator.AddLayer([
      TNNetInput.Create(32,32,3),
      TNNetConvolutionLinear.Create(64 * NeuronMultiplier,3,1,2,0), // downsample to 16x16
      TNNetSELU.Create(),
      TNNetConvolutionLinear.Create(64 * NeuronMultiplier,3,1,2,0), // downsample to 8x8
      TNNetSELU.Create(),
      TNNetConvolutionLinear.Create(64 * NeuronMultiplier,3,1,2,0), // downsample to 4x4
      TNNetSELU.Create(),
//      TNNetDropout.Create(0.4),
      TNNetFullConnectLinear.Create(2),
      TNNetSoftMax.Create()
    ]);*)
    FDiscriminator.AddLayer49([
      TNNetInput.Create4(32,32,3),
      TNNetConvolutionReLU.Create(64 * NeuronMultiplier,3,0,1,0), // downsample to 15x15
      //constructor Create(pPoolSize: integer; pStride:integer = 0; pPadding: integer = 0); overload;
      TNNetMaxPool.Create44(2,0,0),
      TNNetConvolutionReLU.Create(64 * NeuronMultiplier,3,0,1,0), // downsample to 7x7
      TNNetMaxPool.Create44(2,0,0),
      TNNetConvolutionReLU.Create(128 * NeuronMultiplier,3,0,1,0), // downsample to 3x3
      TNNetDropout.Create12(0.5,1),
      TNNetMaxPool.Create44(2,0,0),
      //constructor Create(pSize: integer; pSuppressBias: integer = 0); overload;
      //RegisterMethod('Constructor Create( pSizeX, pSizeY, pDepth : integer; pSuppressBias : integer)');
    //RegisterMethod('Constructor Create28( pSize : integer; pSuppressBias : integer);');
      TNNetFullConnectLinear.Create28(128,0),
      TNNetFullConnectLinear.Create28(2,0),
      TNNetSoftMax.Create()
    ]);
   //check index 0
   // function GetFirstImageNeuronalLayerIdx(FromLayerIdx:integer = 0): integer; {$IFDEF Release} inline; {$ENDIF}
  FDiscriminator.Layers[FDiscriminator.GetFirstImageNeuronalLayerIdx(0)].InitBasicPatterns();
  end else begin
    WriteLn('Loading discriminator.');
    FDiscriminator.LoadFromFile(FBaseName+'discriminator.nn');
    showmessagebig('discriminator loaded_ ');
    TNNetInput(FDiscriminator.Layers[0]).EnableErrorCollection;
    FDiscriminator.DebugStructure();
    FDiscriminator.DebugWeights();
  end;
  FDiscriminator.DebugStructure();

  FDiscriminatorClone := FDiscriminator.Clone();
  TNNetInput(FDiscriminatorClone.Layers[0]).EnableErrorCollection;

  FFit.EnableClassComparison();
  FFit.OnAfterEpoch := @{Self.}DiscriminatorOnAfterEpoch;
  FFit.OnAfterStep := @{Self.}DiscriminatorOnAfterStep;
  FFit.OnStart := @{Self.}DiscriminatorOnStart;
  FFit.DataAugmentationFn := @{Self.}DiscriminatorAugmentation;
  FFit.LearningRateDecay := 0.00001;
  FFit.AvgWeightEpochCount := 1;
  FFit.InitialLearningRate := 0.001;
  FFit.FileNameBase := FBaseName+'GenerativeNeuralAPI';
  {$ifdef OpenCL}
  if FHasOpenCL then begin
    FFit.EnableOpenCL(FEasyOpenCL.PlatformIds[0], FEasyOpenCL.Devices[0]);
    FGenerative.EnableOpenCL(FEasyOpenCL.PlatformIds[0], FEasyOpenCL.Devices[0]);
  end;
  {$endif}
  //Debug only: FFit.MaxThreadNum := 1;
  //FFit.FitLoading(FDiscriminator, 64*10, 500, 500, 64, 35000, 
    //       @GetDiscriminatorTrainingPair, nil, nil); // This line does the same as below
  //FFit.FitLoading1(FDiscriminator, 64*10, 500, 500, 64, 35000, 
    //            @GetDiscriminatorTrainingProc, nil, nil);  //}
         // This line does the same as above
  {FFit.FitLoading1(Nil, 64*10, 500, 500, 64, 35000, 
                    Nil, nil, nil);  //}  
  //FFit.runtestbatch(20)                  
  //FFit.FitLoading2(FDiscriminator, 64*10, 500, 500, 64, 35000, 
    //              @GetDiscriminatorTrainingPair, nil, nil); // This 
  FGeneratives:= TNNetDataParallelism.Create74(FGenerative, FFit.MaxThreadNum, true);
  {$ifdef OpenCL}
  if FHasOpenCL then begin
    FGeneratives.EnableOpenCL(FEasyOpenCL.PlatformIds[0], FEasyOpenCL.Devices[0]);
  end;
  {$endif}
  // is an onstart event!
  //FGeneratives:= TNNetDataParallelism.Create74(FGenerative, FFit.MaxThreadNum, true);
  //BuildTrainingPairs();  
  FFit.FitLoading22(FDiscriminator, 64*10, 500, 500, 64, 35000, 
                @GetDiscriminatorTrainingProc, nil, nil);  //}                                      
    //FFit.FitLoading(FDiscriminator, 64*10, 500, 500, 64, 35000,  @GetDiscriminatorTrainingPair, nil, nil); // This line does the same as below
 
  if Assigned(FGeneratives) then begin
  //FreeAndNil(FGeneratives);
     FGeneratives.Free;
     FGeneratives:= Nil;
  end;   
  FGenerative.Free;
  FDiscriminator.Free;
  FDiscriminatorClone.Free;
end;

function GetDiscriminatorTrainingPair(Idx: integer; ThreadId: integer): TNNetVolumePair;
var
  RandomValue, RandomPos: integer;
  LocalPair: TNNetVolumePair;
begin
  if (FRealPairs.Count = 0) then begin
    WriteLn('Error: discriminator real pairs have no element');
    Result := nil;
    exit;
  end;
  if FGeneratedPairs.Count = 0 then begin
    WriteLn('Error: discriminator generated/fake pairs have no element');
    Result := nil;
    exit;
  end;

  RandomValue := Random(1000);
  if RandomValue < 500 then begin
    RandomPos := Random(FRealPairs.Count);
    Result := FRealPairs[RandomPos];
    Result.O.SetClassForSoftMax(1);
    if Result.I.Size <> 32*32*3 then begin
      //WriteLn('ERROR: Real Pair index ',RandomPos,'has wrong size:', Result.I.Size);
      WriteLn('ERROR: Real Pair index '+itoa(RandomPos)+
            'has wrong size:'+itoa( Result.I.Size));
    end;
    // Debug Only: if (Random(100)=0) then DisplayInputImage(Result.I, FColorEncoding);
  end else begin
    LocalPair := FGeneratedPairs[ThreadId];
    LocalPair.I.Resize(FRandomSizeX, FRandomSizeY, FRandomDepth);
    //procedure Randomize(a:integer=10000; b:integer=5000; c:integer=5000); 
    LocalPair.I.Randomize(10000,5000,5000);
    LocalPair.I.NormalizeMax(2);
    //procedure Compute(pInput: TNNetVolume; FromLayerIdx:integer = 0); overload;
    FGeneratives[ThreadId].Compute65(LocalPair.I,0);
    FGeneratives[ThreadId].GetOutput(LocalPair.I);
    Result := LocalPair;
    Result.O.SetClassForSoftMax(0);
    if Result.I.Size <> 32*32*3 then begin
      WriteLn('ERROR: Generated Pair has wrong size:'+itoa( Result.I.Size));
    end;
  end;
end;

procedure GetDiscriminatorTrainingProc(Idx: integer;
                            ThreadId: integer; pInput, pOutput: TNNetVolume);
var
  LocalPair: TNNetVolumePair;
begin
  writeln('___index is__ '+itoa(idx));
  LocalPair := GetDiscriminatorTrainingPair(Idx, ThreadId);
  pInput.Copy(LocalPair.I);
  pOutput.Copy(LocalPair.O);
end;

procedure DiscriminatorOnAfterEpoch(Sender: TObject);
var
  LoopCnt, MaxLoop: integer;
  ExpectedDiscriminatorOutput, Transitory, DiscriminatorFound, GenerativeInput: TNNetVolume;
  Error: TNeuralFloat;
begin
  if (FFit.TrainingAccuracy <= 0.745) or FFit.ShouldQuit
  then exit;
  WriteLn('Training Generative Start.');
  //constructor Create(pSizeX, pSizeY, pDepth: integer; c: T = 0); {$IFNDEF FPC} overload; {$ENDIF}
  ExpectedDiscriminatorOutput := TNNetVolume.Create0(2, 1, 1,0);
  ExpectedDiscriminatorOutput.SetClassForSoftMax(1);
  DiscriminatorFound := TNNetVolume.Create3(ExpectedDiscriminatorOutput);
  Transitory := TNNetVolume.Create3(FDiscriminatorClone.Layers[0].OutputError);
  GenerativeInput := TNNetVolume.Create0(FRandomSizeX, FRandomSizeY, FRandomDepth,0);
  FDiscriminatorClone.CopyWeights(FDiscriminator);
  FGenerative.SetBatchUpdate(true);
  FGenerative.SetLearningRate(FFit.CurrentLearningRate*FLearningRateProportion,0);
  FGenerative.SetL2Decay(0.00001);
  FDiscriminatorClone.SetBatchUpdate(true);
  FDiscriminatorClone.SetL2Decay(0.0);
  MaxLoop := Round(100 * (1/FLearningRateProportion));
  begin
    Error := 0;
    FDiscriminatorClone.RefreshDropoutMask();
    for LoopCnt := 1 to MaxLoop do begin
      if FFit.ShouldQuit then break;
      FDiscriminatorClone.ClearDeltas();
      FDiscriminatorClone.ClearInertia();
      //Randomize(a:integer=10000; b:integer=5000; c:integer=5000); {$IFDEF Release} inline; {$ENDIF}
      GenerativeInput.Randomize(10000,5000,5000);
      GenerativeInput.NormalizeMax(2);
      if FGenerative.Layers[0].Output.Size<>GenerativeInput.Size then begin
        Write('.');
        FGenerative.Layers[0].Output.ReSize10(GenerativeInput);
        FGenerative.Layers[0].OutputError.ReSize10(GenerativeInput);
      end;
      FGenerative.Compute65(GenerativeInput,0);
      FGenerative.GetOutput(Transitory);
      //procedure Compute(pInput: TNNetVolume; FromLayerIdx:integer = 0); overload;
      FDiscriminatorClone.Compute65(Transitory,0);
      FDiscriminatorClone.GetOutput(DiscriminatorFound);
      FDiscriminatorClone.Backpropagate69(ExpectedDiscriminatorOutput);
      Error:= Error + ExpectedDiscriminatorOutput.SumDiff(DiscriminatorFound);
      Transitory.Sub64(FDiscriminatorClone.Layers[0].OutputError);
      FGenerative.Backpropagate69(Transitory);
      FGenerative.NormalizeMaxAbsoluteDelta(0.001);
      FGenerative.UpdateWeights();
      if LoopCnt mod 10 = 0 then TFormVisualLearningProcessMessages();
      if LoopCnt mod 100 = 0 then DisplayInputImage(Transitory, FColorEncoding);
    end;
    FDiscriminatorClone.Layers[0].OutputError.PrintDebug();
    WriteLn('Clone.Layers end');
    WriteLn('Generative error:'+flots(Error)); //r:6:4);
  end;
  //Debug:
  //FGenerative.DebugErrors();
  //FGenerative.DebugWeights();
  //FDiscriminatorClone.DebugWeights();
  FGeneratives.CopyWeights(FGenerative);
  GenerativeInput.Free;
  ExpectedDiscriminatorOutput.Free;
  Transitory.Free;
  DiscriminatorFound.Free;
  if FFit.CurrentEpoch mod 100 = 0 then begin
    WriteLn('Saving '+ FBaseName);
    FGenerative.SaveToFile(FBaseName+'generative.nn');
    FDiscriminator.SaveToFile(FBaseName+'discriminator.nn');
    TFormVisualLearningSaveScreenshot(FBaseName+'cai-neural-gan.bmp');
  end;
  WriteLn('Training Generative Finish with:'+flots( Error)); //:6:4);;
  //DisplayInputImage(FRealPairs[Random(FRealPairs.Count)].I, FColorEncoding);
end;

procedure DiscriminatorOnAfterStep(Sender: TObject);
begin
  LabClassRate.Caption := StrPadLeft(IntToStr(Round(FFit.TrainingAccuracy*100))+
              +'%',4,'-');
  TFormVisualLearningProcessMessages();
end;

procedure DiscriminatorAugmentation(pInput: TNNetVolume;
  ThreadId: integer);
begin
  if Random(1000)>500 then pInput.FlipX();
  //if Random(1000)>750 then pInput.MakeGray(FColorEncoding);
end;

procedure TFormVisualLearningSaveScreenshot(filename: string);
begin
  try
    WriteLn(' Saving '+filename+'.');
    SaveHandleToBitmap(filename, FormVisualLearning.Handle);
  except
    // Nothing can be done if this fails.
  end;
end;

procedure BuildTrainingPairs();
var
  FakePairCnt: integer;
  ImgTrainingVolume: TNNetVolume;
  DiscriminatorOutput, GenerativeOutput: TNNetVolume;
begin
  DiscriminatorOutput := TNNetVolume.Create0(2, 1, 1,0);
  GenerativeOutput := TNNetVolume.Create0(32, 32, 3, 0);
  if FRealPairs.Count = 0 then begin
    //for ImgTrainingVolume in ImgTrainingVolumes do
    //for it:= low(imgTrainingVolumes) to high(imgTrainingVolumes) do   //fix
     for it:= 0 to imgTrainingVolumes.count-1 do   //fix
    begin
      ImgTrainingVolume:= imgTrainingVolumes[it];
      if ImgTrainingVolume.Tag = 5 then begin//5
        DiscriminatorOutput.SetClassForSoftMax(1);
        FRealPairs.Add
        (
        //constructor CreateCopying(pA, pB: TNNetVolume); overload;
          TNNetVolumePair.CreateCopying83
          (
            ImgTrainingVolume,
            DiscriminatorOutput
          )
        );
        //Debug only: if (Random(100)=0) then DisplayInputImage(ImgTrainingVolume, FColorEncoding);
      end
    end;
  end;

  DiscriminatorOutput.SetClassForSoftMax(0);

  if FGeneratedPairs.Count < FFit.MaxThreadNum then begin
    for FakePairCnt := 1 to FFit.MaxThreadNum do begin
      FGeneratedPairs.Add
      (
        TNNetVolumePair.CreateCopying83
        (
          GenerativeOutput,
          DiscriminatorOutput
        )
      );
    end;
  end;    //*)
  ImgTrainingVolumes.Clear;
  ImgValidationVolumes.Clear;                        
  ImgTestVolumes.Clear;
  GenerativeOutput.Free;
  DiscriminatorOutput.Free;
end;

procedure loadvisualneuralForm;
begin
FormVisualLearning:= TFormVisualLearning.create(self);
ImgSample:= TImage.create(self);
//imgdirect:= TImage.create(self);
with  FormVisualLearning do begin
  Left:= 665
  Height:= 605
  Top:= 141
  Width:= 1133
  Caption:= 'CAI Art - CIFAR-10 based Generative Adversarial Network - maXbox4';
  ClientHeight := 605
  ClientWidth := 1133
  //DesignTimePPI := 120
  OnClose:= @TFormVisualLearningFormClose;
  OnCreate:= @TFormVisualLearningFormCreate;
  OnDestroy:= @TFormVisualLearningFormDestroy;
  icon.loadfromresourcename(hinstance,'XARCHIPELAGO');
  //show;
  //TFormVisualLearningFormCreate(Self);
  ShoW;
  Position := poScreenCenter
  //LCLVersion := '2.0.2.0'
  ButLearn:= TButton.create(self)
  with butlearn do begin
   parent:= formvisuallearning;
    Left := 768
    Height := 45
    Top := 8
    Width := 168
    Caption := 'Start'
    Enabled := True;
    OnClick := @TFormVisualLearningButLearnClick;
    ParentFont := False
    TabOrder := 0
  end;
  //ImgSample:= TImage.create(self)
  with imgsample do begin
   parent:= formvisuallearning;
    Left := 1080
    Height := 32
    Top := 8
    Width := 32
    Stretch := True
  end;
 //imgdirect:= TImage.create(self);
 { with imgdirect do begin
   parent:= formvisuallearning;
    Left := 545
    Height := 142
    Top := 120
    Width := 142
    Stretch := True
  end;   }
  ChkRunOnGPU:= TCheckBox.create(self)
  with ChkRunOnGPU do begin
   parent:= formvisuallearning;
    Left := 136
    Height := 24
    Top := 40
    Width := 102
    Caption := 'Run on GPU'
    TabOrder := 5
  end ;
 labClassRate:=TLabel.create(self);
 with lABclassrate do begin
  parent:= formvisuallearning;
    Left := 8
    Height := 46
    Top := 8
    Width := 96
    Caption := '  0%'
    Color := 15526549
    Font.CharSet := ANSI_CHARSET
    Font.Height := -40
    Font.Name := 'Courier New'
    //Font.Pitch := fpFixed;
    //Font.Quality := fqDraft
    Font.Style := [fsBold]
    ParentColor := False
    ParentFont := False
    Transparent := False
  end ;
  LabComplexity:= TLabel.create(self)
  with labcomplexity do begin
   parent:= formvisuallearning;
    Left := 472
    Height := 20
    Top := 12
    Width := 78
    Caption := 'Learning Rate:'
    ParentColor := False
  end;
  GrBoxNeurons:= TGroupBox.create(self)
  with grboxneurons do begin
   parent:= formvisuallearning;
    Left := 8
    Height := 504
    Top := 73
    Width := 344
    ParentFont := False
    TabOrder := 1
  end;
  TFormVisualLearningFormCreate(Self);
  ChkBigNetwork:= TCheckBox.create(self)
  with ChkBigNetwork do begin
  parent:= formvisuallearning;
    Left := 136
    Height := 24
    Top := 8
    Width := 259
    Caption := 'Bigger (and slower) neural network.'
    TabOrder := 2
  end;
   RadRGB:= TRadioButton.create(self)
   with radrgb do begin
    parent:= formvisuallearning;
    Left := 688
    Height := 24
    Top := 8
    Width := 52
    Caption := 'RGB'
    Checked := True
    TabOrder := 4
    TabStop := True
  end;
  RadLAB:= TRadioButton.create(self);
  with radlab do begin
   parent:= formvisuallearning;
    Left := 688
    Height := 24
    Top := 40
    Width := 50
    Caption := 'LAB'
    TabOrder := 3
  end;
  //object ButLoadFile: TButton
  {ButLoadFile:= TButton.create(self)
  with ButLoadFile do begin
   parent:= formvisuallearning;  
    Left := 30
    Height := 30
    Top := 16
    Width := 160
    Caption := 'Load Neural Network...'
    OnClick := @ButLoadFileClick
    ParentFont := False
    TabOrder := 2
  end; }
  LabComplexity:= TLabel.create(self)
  with LabComplexity do begin
   parent:= formvisuallearning;
    Left := 472
    Height := 20
    Top := 40
    Width := 78
    Caption := 'Complexity Layer:'
    ParentColor := False
  end ;
  ComboComplexity:= TComboBox.create(self)
  with ComboComplexity do begin
   parent:= formvisuallearning;
    Left := 576
    Height := 28
    Top := 40
    Width := 80
    //Enabled := False    $
    ItemIndex := 2
    Items.add('3');
      Items.add('4');
      Items.add('8');
      Items.add('16');
      Items.add('32');
      Items.add('64');
      Items.add('128');
      Items.add('256');
    //)
    ItemHeight := 20
    Style := csDropDownList
    TabOrder := 3
    ItemIndex := 2;
  end ;
  ComboLearningRate:= TComboBox.create(self)
  with ComboLearningRate do begin
   parent:= formvisuallearning;
    Left := 576
    Height := 28
    Top := 8
    Width := 80
    ItemHeight := 20
    ItemIndex := 0
    Items.add ('High');
    items.add('Normal');
    items.add('Low');
    Text := 'High'
    //Checked := True
    //State := cbChecked
    Style := csDropDownList;
    TabOrder := 7
    ItemIndex := 0;
  end;
  //object ChkForceInputRange: TCheckBox
 (* ChkForceInputRange:= TCheckBox.create(self)
  with ChkForceInputRange do begin
   parent:= formvisuallearning;
    Left := 768
    Height := 24
    Top := 32
    Width := 144
    Caption := 'Force Input Range'
    TabOrder := 5
  end;
  OpenDialogNN:= TOpenDialog.create(self);
  with opendialogNN do begin
    Title := 'Open existing Neural Network File'
    Filter := 'Neural Network|*.nn'
    //left := 488
    //top := 120
  end;  *)
 end;
end; 


procedure TFormVisualLearningProcessMessages();
begin
  Application.ProcessMessages();
end;

begin //@main

  //TFormVisualLearningFormCreate(self);
  csLearningRates[0]:= 1; csLearningRates[1]:= 0.1;
  csLearningRates[2]:= 0.01;
  loadvisualneuralForm;

End.
end.

log file:
CheckCIFARFile: TRUE
Loading 10K images from file "data_batch_1.bin" ...
 GLOBAL MIN MAX -2  1.984375
 Done...
Loading 10K images from file "data_batch_2.bin" ...
 GLOBAL MIN MAX -2  1.984375
 Done...
Loading 10K images from file "data_batch_3.bin" ...
 GLOBAL MIN MAX -2  1.984375
 Done...
Loading 10K images from file "data_batch_4.bin" ...
 GLOBAL MIN MAX -2  1.984375
 Done...
Loading 10K images from file "data_batch_5.bin" ...
 GLOBAL MIN MAX -2  1.984375
 Done...
Loading 10K images from file "test_batch.bin" ...
 GLOBAL MIN MAX -2  1.984375
 Done...
Creating Neural Networks...
Creating generative.
 Layers: 22
 Neurons: 555
 Weights: 554864
 Sum: 12.93408203125
Layer 0 Neurons:0 Weights:0 TNNetInput(4,4,8,0,0) Output:4,4,8 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  0.0000 
 Branches:1
Layer 1 Neurons:128 Weights:9216 TNNetConvolutionReLU(128,3,1,1,0) Output:4,4,128 Learning Rate:0.0100  Inertia:0.90  Weight Sum: -1.0796 
 Parent:0
 Branches:1
Layer 2 Neurons:1 Weights:2 TNNetMovingStdNormalization(0,0,0,0,0) Output:4,4,128 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  1.0000 
 Parent:1
 Branches:1
Layer 3 Neurons:128 Weights:147456 TNNetConvolutionReLU(128,3,1,1,0) Output:4,4,128 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  8.3919 
 Parent:2
 Branches:1
Layer 4 Neurons:1 Weights:2 TNNetMovingStdNormalization(0,0,0,0,0) Output:4,4,128 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  1.0000 
 Parent:3
 Branches:1
Layer 5 Neurons:0 Weights:0 TNNetDeMaxPool(2,2,0,0,0) Output:8,8,128 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  0.0000 
 Parent:4
 Branches:1
Layer 6 Neurons:64 Weights:204800 TNNetConvolutionReLU(64,5,2,1,0) Output:8,8,64 Learning Rate:0.0100  Inertia:0.90  Weight Sum:-15.2941 
 Parent:5
 Branches:1
Layer 7 Neurons:1 Weights:2 TNNetMovingStdNormalization(0,0,0,0,0) Output:8,8,64 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  1.0000 
 Parent:6
 Branches:1
Layer 8 Neurons:64 Weights:36864 TNNetConvolutionReLU(64,3,1,1,0) Output:8,8,64 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  5.1947 
 Parent:7
 Branches:1
Layer 9 Neurons:1 Weights:2 TNNetMovingStdNormalization(0,0,0,0,0) Output:8,8,64 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  1.0000 
 Parent:8
 Branches:1
Layer 10 Neurons:0 Weights:0 TNNetDeMaxPool(2,2,0,0,0) Output:16,16,64 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  0.0000 
 Parent:9
 Branches:1
Layer 11 Neurons:64 Weights:102400 TNNetConvolutionReLU(64,5,2,1,0) Output:16,16,64 Learning Rate:0.0100  Inertia:0.90  Weight Sum: -3.4008 
 Parent:10
 Branches:1
Layer 12 Neurons:1 Weights:2 TNNetMovingStdNormalization(0,0,0,0,0) Output:16,16,64 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  1.0000 
 Parent:11
 Branches:1
Layer 13 Neurons:32 Weights:18432 TNNetConvolutionReLU(32,3,1,1,0) Output:16,16,32 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  2.0497 
 Parent:12
 Branches:1
Layer 14 Neurons:1 Weights:2 TNNetMovingStdNormalization(0,0,0,0,0) Output:16,16,32 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  1.0000 
 Parent:13
 Branches:1
Layer 15 Neurons:0 Weights:0 TNNetDeMaxPool(2,2,0,0,0) Output:32,32,32 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  0.0000 
 Parent:14
 Branches:1
Layer 16 Neurons:32 Weights:25600 TNNetConvolutionReLU(32,5,2,1,0) Output:32,32,32 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  2.1512 
 Parent:15
 Branches:1
Layer 17 Neurons:1 Weights:2 TNNetMovingStdNormalization(0,0,0,0,0) Output:32,32,32 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  1.0000 
 Parent:16
 Branches:1
Layer 18 Neurons:32 Weights:9216 TNNetConvolutionReLU(32,3,1,1,0) Output:32,32,32 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  3.0615 
 Parent:17
 Branches:1
Layer 19 Neurons:1 Weights:2 TNNetMovingStdNormalization(0,0,0,0,0) Output:32,32,32 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  1.0000 
 Parent:18
 Branches:1
Layer 20 Neurons:3 Weights:864 TNNetConvolutionLinear(3,3,1,1,0) Output:32,32,3 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  3.8596 
 Parent:19
 Branches:1
Layer 21 Neurons:0 Weights:0 TNNetReLUL(-40,40,0,0,0) Output:32,32,3 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  0.0000 
 Parent:20
 Branches:0
Creating discriminator.
 Layers: 11
 Neurons: 386
 Weights: 260032
 Sum: 1.34475159645081
Layer 0 Neurons:0 Weights:0 TNNetInput(32,32,3,0,0) Output:32,32,3 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  0.0000 
 Branches:1
Layer 1 Neurons:64 Weights:1728 TNNetConvolutionReLU(64,3,0,1,0) Output:30,30,64 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  2.5174 
 Parent:0
 Branches:1
Layer 2 Neurons:0 Weights:0 TNNetPoolBase(2,2,0,0,0) Output:15,15,64 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  0.0000 
 Parent:1
 Branches:1
Layer 3 Neurons:64 Weights:36864 TNNetConvolutionReLU(64,3,0,1,0) Output:13,13,64 Learning Rate:0.0100  Inertia:0.90  Weight Sum: -2.8658 
 Parent:2
 Branches:1
Layer 4 Neurons:0 Weights:0 TNNetPoolBase(2,2,0,0,0) Output:7,7,64 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  0.0000 
 Parent:3
 Branches:1
Layer 5 Neurons:128 Weights:73728 TNNetConvolutionReLU(128,3,0,1,0) Output:5,5,128 Learning Rate:0.0100  Inertia:0.90  Weight Sum: -4.3569 
 Parent:4
 Branches:1
Layer 6 Neurons:0 Weights:0 TNNetDropout(2,1,0,0,0) Output:5,5,128 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  0.0000 
 Parent:5
 Branches:1
Layer 7 Neurons:0 Weights:0 TNNetPoolBase(2,2,0,0,0) Output:3,3,128 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  0.0000 
 Parent:6
 Branches:1
Layer 8 Neurons:128 Weights:147456 TNNetFullConnectLinear(128,1,1,0,0) Output:128,1,1 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  3.6752 
 Parent:7
 Branches:1
Layer 9 Neurons:2 Weights:256 TNNetFullConnectLinear(2,1,1,0,0) Output:2,1,1 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  2.3749 
 Parent:8
 Branches:1
Layer 10 Neurons:0 Weights:0 TNNetSoftMax(0,0,0,0,0) Output:2,1,1 Learning Rate:0.0100  Inertia:0.90  Weight Sum:  0.0000 
 Parent:9
 Branches:0
Debug SavetokenStructureToString: 
Debug structuretostring: -1)TNNetInput:32;32;3;0;0;0;0;0#0)TNNetConvolutionReLU:64;3;0;1;0;0;0;0#1)TNNetPoolBase:2;2;0;0;0;0;0;0#2)TNNetConvolutionReLU:64;3;0;1;0;0;0;0#3)TNNetPoolBase:2;2;0;0;0;0;0;0#4)TNNetConvolutionReLU:128;3;0;1;0;0;0;0#5)TNNetDropout:2;1;0;0;0;0;0;0#6)TNNetPoolBase:2;2;0;0;0;0;0;0#7)TNNetFullConnectLinear:128;1;1;0;0;0;0;0#8)TNNetFullConnectLinear:2;1;1;0;0;0;0;0#9)TNNetSoftMax:0;0;0;0;0;0;0;0
Debug TNNet.Struct.LoadFromString ST: 
-1)TNNetInput:32;32;3;0;0;0;0;0
0)TNNetConvolutionReLU:64;3;0;1;0;0;0;0
1)TNNetPoolBase:2;2;0;0;0;0;0;0
2)TNNetConvolutionReLU:64;3;0;1;0;0;0;0
3)TNNetPoolBase:2;2;0;0;0;0;0;0
4)TNNetConvolutionReLU:128;3;0;1;0;0;0;0
5)TNNetDropout:2;1;0;0;0;0;0;0
6)TNNetPoolBase:2;2;0;0;0;0;0;0
7)TNNetFullConnectLinear:128;1;1;0;0;0;0;0
8)TNNetFullConnectLinear:2;1;1;0;0;0;0;0
9)TNNetSoftMax:0;0;0;0;0;0;0;0

debug createLayer():TNNetInput-addLayerAfter-1...TNNetInput:32;32;3;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter0...TNNetConvolutionReLU:64;3;0;1;0;0;0;0
debug createLayer():TNNetPoolBase-addLayerAfter1...TNNetPoolBase:2;2;0;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter2...TNNetConvolutionReLU:64;3;0;1;0;0;0;0
debug createLayer():TNNetPoolBase-addLayerAfter3...TNNetPoolBase:2;2;0;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter4...TNNetConvolutionReLU:128;3;0;1;0;0;0;0
debug createLayer():TNNetDropout-addLayerAfter5...TNNetDropout:2;1;0;0;0;0;0;0
debug createLayer():TNNetPoolBase-addLayerAfter6...TNNetPoolBase:2;2;0;0;0;0;0;0
debug createLayer():TNNetFullConnectLinear-addLayerAfter7...TNNetFullConnectLinear:128;1;1;0;0;0;0;0
debug createLayer():TNNetFullConnectLinear-addLayerAfter8...TNNetFullConnectLinear:2;1;1;0;0;0;0;0
debug createLayer():TNNetSoftMax-addLayerAfter9...TNNetSoftMax:0;0;0;0;0;0;0;0
Debug Struct.LoadFromString: 2
Debug TNNet Loaded Data Layers:11  SCount:11
Debug Data.LoadFromString: 2
Debug SavetokenStructureToString: 
Debug structuretostring: -1)TNNetInput:4;4;8;0;0;0;0;0#0)TNNetConvolutionReLU:128;3;1;1;0;0;0;0#1)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0#2)TNNetConvolutionReLU:128;3;1;1;0;0;0;0#3)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0#4)TNNetDeMaxPool:2;2;0;0;0;0;0;0#5)TNNetConvolutionReLU:64;5;2;1;0;0;0;0#6)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0#7)TNNetConvolutionReLU:64;3;1;1;0;0;0;0#8)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0#9)TNNetDeMaxPool:2;2;0;0;0;0;0;0#10)TNNetConvolutionReLU:64;5;2;1;0;0;0;0#11)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0#12)TNNetConvolutionReLU:32;3;1;1;0;0;0;0#13)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0#14)TNNetDeMaxPool:2;2;0;0;0;0;0;0#15)TNNetConvolutionReLU:32;5;2;1;0;0;0;0#16)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0#17)TNNetConvolutionReLU:32;3;1;1;0;0;0;0#18)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0#19)TNNetConvolutionLinear:3;3;1;1;0;0;0;0#20)TNNetReLUL:-40;40;0;0;0;0;0;0
Debug Fit TNNetDataParallelismCloneLen: 10905271
Debug TNNet.Struct.LoadFromString ST: 
-1)TNNetInput:4;4;8;0;0;0;0;0
0)TNNetConvolutionReLU:128;3;1;1;0;0;0;0
1)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
2)TNNetConvolutionReLU:128;3;1;1;0;0;0;0
3)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
4)TNNetDeMaxPool:2;2;0;0;0;0;0;0
5)TNNetConvolutionReLU:64;5;2;1;0;0;0;0
6)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
7)TNNetConvolutionReLU:64;3;1;1;0;0;0;0
8)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
9)TNNetDeMaxPool:2;2;0;0;0;0;0;0
10)TNNetConvolutionReLU:64;5;2;1;0;0;0;0
11)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
12)TNNetConvolutionReLU:32;3;1;1;0;0;0;0
13)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
14)TNNetDeMaxPool:2;2;0;0;0;0;0;0
15)TNNetConvolutionReLU:32;5;2;1;0;0;0;0
16)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
17)TNNetConvolutionReLU:32;3;1;1;0;0;0;0
18)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
19)TNNetConvolutionLinear:3;3;1;1;0;0;0;0
20)TNNetReLUL:-40;40;0;0;0;0;0;0

debug createLayer():TNNetInput-addLayerAfter-1...TNNetInput:4;4;8;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter0...TNNetConvolutionReLU:128;3;1;1;0;0;0;0
debug createLayer():TNNetMovingStdNormalization-addLayerAfter1...TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter2...TNNetConvolutionReLU:128;3;1;1;0;0;0;0
debug createLayer():TNNetMovingStdNormalization-addLayerAfter3...TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
debug createLayer():TNNetDeMaxPool-addLayerAfter4...TNNetDeMaxPool:2;2;0;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter5...TNNetConvolutionReLU:64;5;2;1;0;0;0;0
debug createLayer():TNNetMovingStdNormalization-addLayerAfter6...TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter7...TNNetConvolutionReLU:64;3;1;1;0;0;0;0
debug createLayer():TNNetMovingStdNormalization-addLayerAfter8...TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
debug createLayer():TNNetDeMaxPool-addLayerAfter9...TNNetDeMaxPool:2;2;0;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter10...TNNetConvolutionReLU:64;5;2;1;0;0;0;0
debug createLayer():TNNetMovingStdNormalization-addLayerAfter11...TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter12...TNNetConvolutionReLU:32;3;1;1;0;0;0;0
debug createLayer():TNNetMovingStdNormalization-addLayerAfter13...TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
debug createLayer():TNNetDeMaxPool-addLayerAfter14...TNNetDeMaxPool:2;2;0;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter15...TNNetConvolutionReLU:32;5;2;1;0;0;0;0
debug createLayer():TNNetMovingStdNormalization-addLayerAfter16...TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter17...TNNetConvolutionReLU:32;3;1;1;0;0;0;0
debug createLayer():TNNetMovingStdNormalization-addLayerAfter18...TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
debug createLayer():TNNetConvolutionLinear-addLayerAfter19...TNNetConvolutionLinear:3;3;1;1;0;0;0;0
debug createLayer():TNNetReLUL-addLayerAfter20...TNNetReLUL:-40;40;0;0;0;0;0;0
Debug Struct.LoadFromString: 2
Debug TNNet Loaded Data Layers:22  SCount:22
Debug Data.LoadFromString: 2
Debug Fit NN.LoadFromString: 22
Debug SavetokenStructureToString: 
Debug structuretostring: -1)TNNetInput:32;32;3;0;0;0;0;0#0)TNNetConvolutionReLU:64;3;0;1;0;0;0;0#1)TNNetPoolBase:2;2;0;0;0;0;0;0#2)TNNetConvolutionReLU:64;3;0;1;0;0;0;0#3)TNNetPoolBase:2;2;0;0;0;0;0;0#4)TNNetConvolutionReLU:128;3;0;1;0;0;0;0#5)TNNetDropout:2;1;0;0;0;0;0;0#6)TNNetPoolBase:2;2;0;0;0;0;0;0#7)TNNetFullConnectLinear:128;1;1;0;0;0;0;0#8)TNNetFullConnectLinear:2;1;1;0;0;0;0;0#9)TNNetSoftMax:0;0;0;0;0;0;0;0
Debug Fit TNNetDataParallelismCloneLen: 5083262
Debug TNNet.Struct.LoadFromString ST: 
-1)TNNetInput:32;32;3;0;0;0;0;0
0)TNNetConvolutionReLU:64;3;0;1;0;0;0;0
1)TNNetPoolBase:2;2;0;0;0;0;0;0
2)TNNetConvolutionReLU:64;3;0;1;0;0;0;0
3)TNNetPoolBase:2;2;0;0;0;0;0;0
4)TNNetConvolutionReLU:128;3;0;1;0;0;0;0
5)TNNetDropout:2;1;0;0;0;0;0;0
6)TNNetPoolBase:2;2;0;0;0;0;0;0
7)TNNetFullConnectLinear:128;1;1;0;0;0;0;0
8)TNNetFullConnectLinear:2;1;1;0;0;0;0;0
9)TNNetSoftMax:0;0;0;0;0;0;0;0

debug createLayer():TNNetInput-addLayerAfter-1...TNNetInput:32;32;3;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter0...TNNetConvolutionReLU:64;3;0;1;0;0;0;0
debug createLayer():TNNetPoolBase-addLayerAfter1...TNNetPoolBase:2;2;0;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter2...TNNetConvolutionReLU:64;3;0;1;0;0;0;0
debug createLayer():TNNetPoolBase-addLayerAfter3...TNNetPoolBase:2;2;0;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter4...TNNetConvolutionReLU:128;3;0;1;0;0;0;0
debug createLayer():TNNetDropout-addLayerAfter5...TNNetDropout:2;1;0;0;0;0;0;0
debug createLayer():TNNetPoolBase-addLayerAfter6...TNNetPoolBase:2;2;0;0;0;0;0;0
debug createLayer():TNNetFullConnectLinear-addLayerAfter7...TNNetFullConnectLinear:128;1;1;0;0;0;0;0
debug createLayer():TNNetFullConnectLinear-addLayerAfter8...TNNetFullConnectLinear:2;1;1;0;0;0;0;0
debug createLayer():TNNetSoftMax-addLayerAfter9...TNNetSoftMax:0;0;0;0;0;0;0;0
Debug Struct.LoadFromString: 2
Debug TNNet Loaded Data Layers:11  SCount:11
Debug Data.LoadFromString: 2
Debug Fit NN.LoadFromString: 11
Debug SavetokenStructureToString: 
Debug structuretostring: -1)TNNetInput:32;32;3;0;0;0;0;0#0)TNNetConvolutionReLU:64;3;0;1;0;0;0;0#1)TNNetPoolBase:2;2;0;0;0;0;0;0#2)TNNetConvolutionReLU:64;3;0;1;0;0;0;0#3)TNNetPoolBase:2;2;0;0;0;0;0;0#4)TNNetConvolutionReLU:128;3;0;1;0;0;0;0#5)TNNetDropout:2;1;0;0;0;0;0;0#6)TNNetPoolBase:2;2;0;0;0;0;0;0#7)TNNetFullConnectLinear:128;1;1;0;0;0;0;0#8)TNNetFullConnectLinear:2;1;1;0;0;0;0;0#9)TNNetSoftMax:0;0;0;0;0;0;0;0
Debug TNNet.Struct.LoadFromString ST: 
-1)TNNetInput:32;32;3;0;0;0;0;0
0)TNNetConvolutionReLU:64;3;0;1;0;0;0;0
1)TNNetPoolBase:2;2;0;0;0;0;0;0
2)TNNetConvolutionReLU:64;3;0;1;0;0;0;0
3)TNNetPoolBase:2;2;0;0;0;0;0;0
4)TNNetConvolutionReLU:128;3;0;1;0;0;0;0
5)TNNetDropout:2;1;0;0;0;0;0;0
6)TNNetPoolBase:2;2;0;0;0;0;0;0
7)TNNetFullConnectLinear:128;1;1;0;0;0;0;0
8)TNNetFullConnectLinear:2;1;1;0;0;0;0;0
9)TNNetSoftMax:0;0;0;0;0;0;0;0

debug createLayer():TNNetInput-addLayerAfter-1...TNNetInput:32;32;3;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter0...TNNetConvolutionReLU:64;3;0;1;0;0;0;0
debug createLayer():TNNetPoolBase-addLayerAfter1...TNNetPoolBase:2;2;0;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter2...TNNetConvolutionReLU:64;3;0;1;0;0;0;0
debug createLayer():TNNetPoolBase-addLayerAfter3...TNNetPoolBase:2;2;0;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter4...TNNetConvolutionReLU:128;3;0;1;0;0;0;0
debug createLayer():TNNetDropout-addLayerAfter5...TNNetDropout:2;1;0;0;0;0;0;0
debug createLayer():TNNetPoolBase-addLayerAfter6...TNNetPoolBase:2;2;0;0;0;0;0;0
debug createLayer():TNNetFullConnectLinear-addLayerAfter7...TNNetFullConnectLinear:128;1;1;0;0;0;0;0
debug createLayer():TNNetFullConnectLinear-addLayerAfter8...TNNetFullConnectLinear:2;1;1;0;0;0;0;0
debug createLayer():TNNetSoftMax-addLayerAfter9...TNNetSoftMax:0;0;0;0;0;0;0;0
Debug Struct.LoadFromString: 2
Debug TNNet Loaded Data Layers:11  SCount:11
Debug Data.LoadFromString: 2
File name is: ART8-1-RGB-GenerativeNeuralAPI
Learning rate:0.001000 L2 decay:0.000000 Inertia:0.900000 Batch size:64 Step size:64 Staircase ephocs:1
Training volumes: 640
Validation volumes: 500
Test volumes: 500
Computing...
DiscriminatorOnStart(Sender: TObject);
Debug SavetokenStructureToString: 
Debug structuretostring: -1)TNNetInput:4;4;8;0;0;0;0;0#0)TNNetConvolutionReLU:128;3;1;1;0;0;0;0#1)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0#2)TNNetConvolutionReLU:128;3;1;1;0;0;0;0#3)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0#4)TNNetDeMaxPool:2;2;0;0;0;0;0;0#5)TNNetConvolutionReLU:64;5;2;1;0;0;0;0#6)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0#7)TNNetConvolutionReLU:64;3;1;1;0;0;0;0#8)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0#9)TNNetDeMaxPool:2;2;0;0;0;0;0;0#10)TNNetConvolutionReLU:64;5;2;1;0;0;0;0#11)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0#12)TNNetConvolutionReLU:32;3;1;1;0;0;0;0#13)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0#14)TNNetDeMaxPool:2;2;0;0;0;0;0;0#15)TNNetConvolutionReLU:32;5;2;1;0;0;0;0#16)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0#17)TNNetConvolutionReLU:32;3;1;1;0;0;0;0#18)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0#19)TNNetConvolutionLinear:3;3;1;1;0;0;0;0#20)TNNetReLUL:-40;40;0;0;0;0;0;0
Debug Fit TNNetDataParallelismCloneLen: 10905271
Debug TNNet.Struct.LoadFromString ST: 
-1)TNNetInput:4;4;8;0;0;0;0;0
0)TNNetConvolutionReLU:128;3;1;1;0;0;0;0
1)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
2)TNNetConvolutionReLU:128;3;1;1;0;0;0;0
3)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
4)TNNetDeMaxPool:2;2;0;0;0;0;0;0
5)TNNetConvolutionReLU:64;5;2;1;0;0;0;0
6)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
7)TNNetConvolutionReLU:64;3;1;1;0;0;0;0
8)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
9)TNNetDeMaxPool:2;2;0;0;0;0;0;0
10)TNNetConvolutionReLU:64;5;2;1;0;0;0;0
11)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
12)TNNetConvolutionReLU:32;3;1;1;0;0;0;0
13)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
14)TNNetDeMaxPool:2;2;0;0;0;0;0;0
15)TNNetConvolutionReLU:32;5;2;1;0;0;0;0
16)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
17)TNNetConvolutionReLU:32;3;1;1;0;0;0;0
18)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
19)TNNetConvolutionLinear:3;3;1;1;0;0;0;0
20)TNNetReLUL:-40;40;0;0;0;0;0;0

debug createLayer():TNNetInput-addLayerAfter-1...TNNetInput:4;4;8;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter0...TNNetConvolutionReLU:128;3;1;1;0;0;0;0
debug createLayer():TNNetMovingStdNormalization-addLayerAfter1...TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter2...TNNetConvolutionReLU:128;3;1;1;0;0;0;0
debug createLayer():TNNetMovingStdNormalization-addLayerAfter3...TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
debug createLayer():TNNetDeMaxPool-addLayerAfter4...TNNetDeMaxPool:2;2;0;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter5...TNNetConvolutionReLU:64;5;2;1;0;0;0;0
debug createLayer():TNNetMovingStdNormalization-addLayerAfter6...TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter7...TNNetConvolutionReLU:64;3;1;1;0;0;0;0
debug createLayer():TNNetMovingStdNormalization-addLayerAfter8...TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
debug createLayer():TNNetDeMaxPool-addLayerAfter9...TNNetDeMaxPool:2;2;0;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter10...TNNetConvolutionReLU:64;5;2;1;0;0;0;0
debug createLayer():TNNetMovingStdNormalization-addLayerAfter11...TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter12...TNNetConvolutionReLU:32;3;1;1;0;0;0;0
debug createLayer():TNNetMovingStdNormalization-addLayerAfter13...TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
debug createLayer():TNNetDeMaxPool-addLayerAfter14...TNNetDeMaxPool:2;2;0;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter15...TNNetConvolutionReLU:32;5;2;1;0;0;0;0
debug createLayer():TNNetMovingStdNormalization-addLayerAfter16...TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter17...TNNetConvolutionReLU:32;3;1;1;0;0;0;0
debug createLayer():TNNetMovingStdNormalization-addLayerAfter18...TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
debug createLayer():TNNetConvolutionLinear-addLayerAfter19...TNNetConvolutionLinear:3;3;1;1;0;0;0;0
debug createLayer():TNNetReLUL-addLayerAfter20...TNNetReLUL:-40;40;0;0;0;0;0;0
Debug Struct.LoadFromString: 2
Debug TNNet Loaded Data Layers:22  SCount:22
Debug Data.LoadFromString: 2
Debug Fit NN.LoadFromString: 22
___index is__ 1
___index is__ 2
___index is__ 3
___index is__ 4
___index is__ 5
___index is__ 6
___index is__ 7
___index is__ 8
___index is__ 9
___index is__ 10
___index is__ 11
___index is__ 12
___index is__ 13
___index is__ 14
___index is__ 15
___index is__ 16
___index is__ 17
___index is__ 18
___index is__ 19
___index is__ 20
___index is__ 21
___index is__ 22
___index is__ 23
___index is__ 24
___index is__ 25
___index is__ 26
___index is__ 27
___index is__ 28
___index is__ 29
___index is__ 30
___index is__ 31
___index is__ 32
___index is__ 33
___index is__ 34
___index is__ 35
___index is__ 36
___index is__ 37
___index is__ 38
___index is__ 39
___index is__ 40
___index is__ 41
___index is__ 42
___index is__ 43
___index is__ 44
___index is__ 45
___index is__ 46
___index is__ 47
___index is__ 48
___index is__ 49
___index is__ 50
___index is__ 51
___index is__ 52
___index is__ 53
___index is__ 54
___index is__ 55
___index is__ 56
___index is__ 57
___index is__ 58
___index is__ 59
___index is__ 60
___index is__ 61
___index is__ 62
___index is__ 63
___index is__ 64
___index is__ 1
___index is__ 2
___index is__ 3
___index is__ 4
___index is__ 5
___index is__ 6
___index is__ 7
___index is__ 8
___index is__ 9
___index is__ 10
___index is__ 11
___index is__ 12
___index is__ 13
___index is__ 14
___index is__ 15
___index is__ 16
___index is__ 17
___index is__ 18
___index is__ 19
___index is__ 20
___index is__ 21
___index is__ 22
___index is__ 23
___index is__ 24
___index is__ 25
___index is__ 26
___index is__ 27
___index is__ 28
___index is__ 29
___index is__ 30
___index is__ 31
___index is__ 32
___index is__ 33
___index is__ 34
___index is__ 35
___index is__ 36
___index is__ 37
___index is__ 38
___index is__ 39
___index is__ 40
___index is__ 41
___index is__ 42
___index is__ 43
___index is__ 44
___index is__ 45
___index is__ 46
___index is__ 47
___index is__ 48
___index is__ 49
___index is__ 50
___index is__ 51
___index is__ 52
___index is__ 53
___index is__ 54
___index is__ 55
___index is__ 56
___index is__ 57
___index is__ 58
___index is__ 59
___index is__ 60
___index is__ 61
___index is__ 62
___index is__ 63
___index is__ 64
___index is__ 1
___index is__ 2
___index is__ 3
___index is__ 4
___index is__ 5
___index is__ 6
___index is__ 7
___index is__ 8
___index is__ 9
___index is__ 10
___index is__ 11
___index is__ 12
___index is__ 13
___index is__ 14
___index is__ 15
___index is__ 16
___index is__ 17
___index is__ 18
___index is__ 19
___index is__ 20
___index is__ 21
___index is__ 22   ...............
1280 Examples seen. Accuracy: 0.6399 Error: 0.31803 Loss: 0.00000 Threads: 1 Forward time: 3.55s Backward time: 0.36s Step time: 98.60s
1920 Examples seen. Accuracy: 0.6743 Error: 0.31574 Loss: 0.00000 Threads: 1 Forward time: 3.22s Backward time: 0.35s Step time: 100.37s
2560 Examples seen. Accuracy: 0.7055 Error: 0.26061 Loss: 0.00000 Threads: 1 Forward time: 3.14s Backward time: 0.31s Step time: 106.12s


