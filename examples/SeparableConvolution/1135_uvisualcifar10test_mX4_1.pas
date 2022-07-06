unit Uvisualcifar10test_mX4_1;

//{$mode objfpc}{$H+}
//https://sourceforge.net/p/cai/svncode/HEAD/tree/trunk/lazarus/experiments/visualCifar10test/uvisualcifar10test.lfm

interface

{uses
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls,
  ExtCtrls, neuralnetwork, neuraldatasetsv, neuralvolumev, neuraldatasets,
  neuralvolume; }
  
Const PReModel_NN = 'SimpleSeparableImageClassifier124_50_2.nn'; 
      //PReModel_NN = 'SimpleSeparableImageClassifier.nn';  
      //PReModel_NN = 'SimpleSeparableImageClassifier124.nn';
      //PReModel_NN = 'SimpleSeparableImageClassifier124_50_3.nn';
      //PReModel_NN = 'ImageClassifierSELU_Tutor89.nn';
type

  { TFormVisualLearning }

  TFormVisualLearning = {class(}TForm;
  var
    ButTest: TButton;
    EdTestBinFile: TEdit;
    ImgSample: TImage;
    LabClassRate: TLabel;
    LabTestFile: TLabel;
    OpenDialogNN: TOpenDialog;
    procedure TFormVisualLearningButTestClick(Sender: TObject);
    procedure TFormVisualLearningFormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure TFormVisualLearningFormCreate(Sender: TObject);
  //private
    { private declarations }
  var
    FRunning: boolean;
    procedure TFormVisualLearningLearn(Sender: TObject);
  //public
    { public declarations }
  //end;

var
  FormVisualLearning: TFormVisualLearning;

implementation
//{$R *.lfm}

{ TFormVisualLearning }

procedure TFormVisualLearningButTestClick(Sender: TObject);
begin
  if not CheckCIFARFile() then writeln('___NO cifar datasets found___'); //exit;

  if (FRunning) then begin
    FRunning := false;
    ButTest.Caption := 'Retest';
    LabClassRate.Caption := '0%';
  end
  else begin
    //if (OpenDialogNN.Execute()) then begin
    OpenDialogNN.filename := Exepath+PReModel_NN; 
      labtestfile.Caption:= 'CIFAR-10 testing file: '+Exepath+PReModel_NN;
      if FileExists(OpenDialogNN.FileName) then begin
        FRunning := true;
        ButTest.Caption := 'Stop';
        TFormVisualLearningLearn(Sender);
        FRunning := false;
        ButTest.Caption := 'Retest';
      end;
    end;
  //end;
end;

procedure TFormVisualLearningFormClose(Sender: TObject;
  var CloseAction: TCloseAction);
begin
  FRunning := false;
  writeln('closed & free action...');
end;

procedure TFormVisualLearningFormCreate(Sender: TObject);
begin
  FRunning := false;
end;

procedure TFormVisualLearningLearn(Sender: TObject);
var
  NN: TNNet;
  I: integer;
  ImgVolumes: TNNetVolumeList;
  Volume: TNNetVolume;
  pOutput, vOutput, vDisplay: TNNetVolume;
  hit, miss: integer;
  NumClasses: integer;
  ErrorSum, LastError: TNeuralFloat;
  startTime, totalTimeSeconds: double;
  aImage, SImage: array of TImage;
  NeuronCount: integer;
  ImgIdx, simgdx, simginc: integer;
  MaxW, MinW: TNeuralFloat;
  fileName: string;
  firstNeuronalLayer: integer;
  Rate, Loss: TNeuralFloat;
begin
  writeln('Creating CNeural Network...');
  ImgVolumes := TNNetVolumeList.Create(true);
  NumClasses := 10;

  fileName:= Exepath+PReModel_NN; //OpenDialogNN.FileName;

  //--------------------------------------------------------------------
  // creates required volumes to store images
  for I:= 0 to 9999 do begin
    Volume := TNNetVolume.Create();
    ImgVolumes.Add(Volume);
  end;
  //--------------------------------------------------------------------

  NN := TNNet.Create();

  writeln('Loading neural network from file: '+fileName);
  NN.LoadFromFile(fileName);
  NN.EnableDropouts(false);
  firstNeuronalLayer := NN.GetFirstNeuronalLayerIdx(0);

  pOutput := TNNetVolume.Create0(NumClasses,1,1,0);   //or 1
  vOutput := TNNetVolume.Create0(NumClasses,1,1,0);
  vDisplay:= TNNetVolume.Create0(NumClasses,1,1,0);

  SetLength(aImage, NN.Layers[firstNeuronalLayer].Neurons.Count);
  SetLength(sImage, NumClasses);

  for NeuronCount := 0 to NN.Layers[firstNeuronalLayer].Neurons.Count- 1 do begin
    aImage[NeuronCount]:= TImage.Create(FormVisualLearning);
    aImage[NeuronCount].Parent := FormVisualLearning;
    aImage[NeuronCount].Width  := 
              NN.Layers[firstNeuronalLayer].Neurons[NeuronCount].Weights.SizeX;
    aImage[NeuronCount].Height := 
              NN.Layers[firstNeuronalLayer].Neurons[NeuronCount].Weights.SizeY;
    aImage[NeuronCount].Top    := (NeuronCount div 12) * 38 + 160;   //120
    aImage[NeuronCount].Left   := (NeuronCount mod 12) * 38 + 60;
    aImage[NeuronCount].Stretch:=true;
  end;
  
  for simgdx:= 0 to NumClasses- 1 do begin
    sImage[simgdx]:= TImage.Create(FormVisualLearning);
    sImage[simgdx].Parent := FormVisualLearning;
    sImage[simgdx].Width  := 32;
    sImage[simgdx].Height := 32
    sImage[simgdx].Top    := (simgdx div 12) * 38 + 460;   //120
    sImage[simgdx].Left   := (simgdx mod 12) * 38 + 60;
    //aImage[simgdx].Stretch:=true;
  end;


  NN.DebugWeights();
  //WriteLn(' Layers: ',  NN.CountLayers()  );
  //WriteLn(' Neurons:',  NN.CountNeurons() );
  WriteLn('Neural CNN network has: ');
  WriteLn(' Layers: '+ itoa(NN.CountLayers()  ));
  WriteLn(' Neurons: '+ itoa(NN.CountNeurons() ));
  WriteLn(' Weights: '+ itoa(NN.CountWeights() ));
  WriteLn('Computing...');

  begin
    hit  := 0;
    miss := 0;
    ErrorSum := 0;
    LastError := 0;
    startTime := Now();
    simginc:= 0;
    //loadCifar10Dataset(ImgVolumes, EdTestBinFile.Text);  //"data_batch_5.bin"
    loadCifar10Dataset2(ImgVolumes, 5, 0, csEncodeRGB);   //test batch file
    WriteLn(' Totalsize: '+ itoa(ImgVolumes.gettotalsize));
    for I := 0 to ImgVolumes.Count - 1 do begin
      if not(FRunning) then Break;
      ImgIdx := Random(ImgVolumes.Count);
      //-- CAREFUL
      //procedure Compute(pInput: TNNetVolume; FromLayerIdx:integer = 0); overload;
      NN.Compute65(ImgVolumes[ImgIdx],0);
      NN.GetOutput(pOutput);

      vOutput.SetClassForReLU( ImgVolumes[ImgIdx].Tag ); // ReLU - no softmax
      //ErrorSum += vOutput.SumDiff(pOutput);
      ErrorSum:= ErrorSum +vOutput.SumDiff(pOutput);

      if I mod 1000 = 0 then begin
        vDisplay.Copy(ImgVolumes[ImgIdx]);
        vDisplay.Mul26(64);
        vDisplay.Add13(128);

        LoadVolumeIntoTImage(vDisplay, ImgSample, csEncodeRGB);
        ImgSample.Width := 64+40;
        ImgSample.Height := 64+40;
        //ImgSample.Stretch:= True;
        //sImage[I].assign(imgsample);
        sImage[simginc].picture:= imgsample.picture;
        inc(simginc);

        //inc(firstNeuronalLayer);
        for NeuronCount:= 0 to NN.Layers[firstNeuronalLayer].Neurons.Count - 1 do begin
          MaxW:= NN.Layers[firstNeuronalLayer].Neurons[NeuronCount].Weights.GetMax();
          MinW:= NN.Layers[firstNeuronalLayer].Neurons[NeuronCount].Weights.GetMin();
          vDisplay.Copy(NN.Layers[firstNeuronalLayer].Neurons[NeuronCount].Weights);
          vDisplay.Mul26(256/(MaxW-MinW));
          vDisplay.Add13(128);
          LoadVolumeIntoTImage(vDisplay, aImage[NeuronCount],csEncodeRGB);
          aImage[NeuronCount].Width := 35;
          aImage[NeuronCount].Height:= 35;
          //aImage[NeuronCount].Stretch:= True;
        end;
        Application.ProcessMessages();
      end;

      Application.ProcessMessages();

      if pOutput.GetClass() = ImgVolumes[ImgIdx].Tag then begin
        Inc(Hit);
        //WriteLn(' Tag Label: '+ itoa(ImgVolumes[ImgIdx].Tag));
      end else begin
        Inc(Miss);
      end;

      if (Hit>0) and (I>0) and ((I+1) mod 1000 = 0) then begin
        totalTimeSeconds:= (Now() - startTime) * 24 * 60 * 60;
        WriteLn
        (
         { I+1, ' Accuracy:'+ Hit/(Hit+Miss):6:4,
          ' Error:', (ErrorSum-LastError):10:5,
          ' Time:', totalTimeSeconds:6:2,'s',
          ' Forward:', (NN.ForwardTime * 24 * 60 * 60):6:2,'s',
          ' Backward:', (NN.BackwardTime * 24 * 60 * 60):6:2,'s' }
          itoa(I+1)+ ' Accuracy:'+ flots(Hit/(Hit+Miss))+
          ' Error:'+flots((ErrorSum-LastError))+
          ' Time:'+flots(totalTimeSeconds)+'s'+
          ' Forward:'+ flots(NN.ForwardTime * 24 * 60 * 60)+'s'+
          ' Backward:'+flots(NN.BackwardTime * 24 * 60 * 60)+'s'  
        );
        NN.ClearTime();

        LabClassRate.Caption:= format('%.2f%% ',[(Hit*100)/(Hit+Miss)]);
        startTime := Now();
        LastError := ErrorSum;
        Application.ProcessMessages;
      end;
    end;

    NN.DebugWeights();
    // This function tests a neural network on the passed ImgVolumes
 {procedure TestBatch
 (
  NN: TNNet; ImgVolumes: TNNetVolumeList; SampleSize: integer;
  out Rate, Loss, ErrorSum: TNeuralFloat
  ); }
    rate:= 0; loss:= 0;
    ErrorSum:= 0;
    TestBatch(NN, ImgVolumes, 1000, rate, loss, ErrorSum);
    writeln('Test batch score: '+Format(' Rate:%.4f, Loss:%.4f, ErrorSum:%.4f ',
                                          [rate, loss, ErrorSum]));
    LabClassRate.Caption:= format('Ø %.2f%% ',[rate*100]);
  end;

  for NeuronCount := Low(aImage) to High(aImage) do begin
    aImage[NeuronCount].Free;
  end;
  for it := Low(sImage) to High(sImage) do begin
    //sImage[it].Free;
  end;

  //LabClassRate.Caption := '0%';
  vDisplay.Free;
  NN.Free;
  vOutput.Free;
  pOutput.Free;
  ImgVolumes.Free;
end;

procedure loadCifarForm;
begin
FormVisualLearning:= TFormVisualLearning.create(self)
with FormVisualLearning do begin
  Left := 2664
  Height:= 616
  Top:= 194
  Width:= 964
  Caption:= 'Visual CIFAR-10 NN Testing maXbox4'
  ClientHeight:= 616
  ClientWidth:= 964
  //DesignTimePPI:= 120
  OnClose:= @TFormVisualLearningFormClose;
  OnCreate:= @TFormVisualLearningFormCreate;
  Position:= poScreenCenter
  icon.loadfromresourcename(hinstance,'XDIRECTX');
  //LCLVersion:= '2.0.2.0'
  Show();
  ButTest:= TButton.create(self)
  with buttest do begin
   parent:= FormVisualLearning;
    Left:= 350
    Height:= 31
    Top:= 100
    Width:= 200
    Caption:= '&Test Neural Network'
    OnClick:= @TFormVisualLearningButTestClick
    ParentFont:= False
    TabOrder:= 0
  end;
  ImgSample:= TImage.create(self)
  with imgsample do begin
   parent:= FormVisualLearning;
    Left:= 15
    Height:= 33
    Top:= 15
    Width:= 34
    Stretch:= True
  end;
  LabTestFile:= TLabel.create(self)
  with labtestfile do begin
   parent:= FormVisualLearning;
    setbounds(350,10,137,20)
    Caption:= 'CIFAR-10 testing file:'
    ParentColor:= False
    ParentFont:= False
  end;
  EdTestBinFile:= TEdit.create(self)
  with  EdTestBinFile do begin
   parent:= FormVisualLearning;
    setbounds(350,50,220,28)
    ParentFont:= False
    TabOrder:= 1
    Text:= 'test_batch.bin'
  end;
  LabClassRate:= TLabel.create(self)
  with labclassrate do begin
   parent:= FormVisualLearning;
    setbounds(123,8,72,57)
    Caption:= '0%'
    Font.CharSet:= ANSI_CHARSET
    Font.Height:= -50
    Font.Name:= 'Arial'
    //Font.Pitch:= fpVariable;
    //Font.Quality:= fqDraft
    ParentColor:= False
    ParentFont:= False
  end;
  OpenDialogNN:= TOpenDialog.create(self);
  with opendialognn do begin
    Title:= 'Open existing Neural Network file'
    Filter:= 'Neural Network|*.nn'
    //left:= 650   //top:= 20
  end;
 end;
end; 

begin //@main

 loadCifarForm();

End.
end.

ref:    csMachineAnimalCifar10Labels: array[0..9] of string =
  (
    'airplane',
    'automobile',
    'ship',  // used to be bird
    'truck', // used to be cat
    'deer',  // used to be deer
    'dog',   // used to be dog
    'frog',  // used to be frog
    'horse', // used to be horse
    'bird',  // used to be ship
    'cat'    // used to be truck
  );  //}
  
   Directory of C:\maXbox\EKON_BASTA\EKON24\cifar-10-batches-bin\resized

28/06/2022  15:50    <DIR>          .
28/06/2022  15:50    <DIR>          ..
28/06/2022  15:50    <DIR>          cifar10-128
28/06/2022  15:47    <DIR>          cifar10-32
28/06/2022  15:50    <DIR>          cifar10-64
28/06/2022  15:50    <DIR>          test
28/06/2022  15:50    <DIR>          train
               0 File(s)              0 bytes
28/06/2022  15:50    <DIR>          class0
28/06/2022  15:50    <DIR>          class1
28/06/2022  15:50    <DIR>          class2
28/06/2022  15:50    <DIR>          class3
28/06/2022  15:50    <DIR>          class4
28/06/2022  15:50    <DIR>          class5
28/06/2022  15:50    <DIR>          class6
28/06/2022  15:50    <DIR>          class7
28/06/2022  15:50    <DIR>          class8
28/06/2022  15:50    <DIR>          class9
