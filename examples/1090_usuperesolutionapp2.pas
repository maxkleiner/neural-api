unit usuperesolutionapp_mX4;
//{$mode objfpc}{$H+}

interface

{uses
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls,
  ExtCtrls, neuraldatasets, neuraldatasetsv, fpimage, IntfGraphics, lcltype, Buttons,
  neuralnetwork, neuralvolume, neuralvolumev;   }

type
  { TForm1 }
  TForm1 = {class(}TForm;
  var
    BitBtn13: TBitBtn;
    CheckPause: TCheckBox;
    Image1: TImage;
    ImageHE: TImage;
    ImageVE: TImage;
    ImageGray: TImage;
    Label1: TLabel;
    Label2: TLabel;
    Label3: TLabel;
    Label4: TLabel;
    Label5: TLabel;
    procedure TForm1BitBtn13Click(Sender: TObject);
    procedure TForm1FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure TForm1FormCreate(Sender: TObject);
  //private
  var
    FWantQuit: boolean;
    function CheckBinFiles: boolean;
  //public

  //end;
  
 function CreateResizingNN(SizeX, SizeY: integer; FileName: string): THistoricalNets;
 procedure LoadResizingWeights(NN:TNNet; FileName:string; FailIfNotFound:boolean{ = false});
 

  { TEvolutionary }
var
  Form1: TForm1;

implementation
//uses usuperresolutionexample;

 //unit usuperresolutionexample;
// need PReModel_NN_Res = '.\examples\super-resolution-7-64-sep.nn';

const
  csExampleBaseFileName = '1090_super-resolution-7-64-sep';
  csExampleFileName {:string} = '1090_super-resolution-7-64-sep.nn';
  csExampleBottleNeck {:integer =} = 16;
  csExampleNeuronCount = 64;
  csExampleLayerCount = 7;
  csExampleIsSeparable {:boolean} =  true;          
  
function CreateResizingNN(SizeX, SizeY: integer; FileName: string): THistoricalNets;
var
  NN: THistoricalNets;
begin
  NN := THistoricalNets.Create();
  NN.AddSuperResolution({pSizeX=}SizeX, {pSizeY=}SizeY,
    {BottleNeck=}csExampleBottleNeck, {pNeurons=}csExampleNeuronCount,
    {pLayerCnt=}csExampleLayerCount, {IsSeparable=}csExampleIsSeparable);
  LoadResizingWeights(NN, FileName, true);
  Result := NN;
end;

procedure LoadResizingWeights(NN:TNNet; FileName:string; FailIfNotFound:boolean {= false});
begin
  if FileExists(FileName) then begin
    NN.LoadDataFromFile(FileName);
    WriteLn('Neural network file found:'+ FileName);
  end else
  if FileExists('./examples/SuperResolution/'+FileName) then begin
    NN.LoadDataFromFile('./examples/SuperResolution/'+FileName);
    WriteLn('Neural network file found at ./examples/SuperResolution : '+ FileName);
  end else begin
    if FailIfNotFound then begin
      WriteLn('ERROR: '+FileName+' can''t be found. Please run SuperResolutionTrain.');
      //ReadLn();
    end;
  end;
end;

var csTinyImageLabel: array[0..9] of string;
procedure setClassifierLabelsTiny;
begin
  csTinyImageLabel[0]:= 'airplane';
  csTinyImageLabel[1]:= 'automobile';
  csTinyImageLabel[2]:= 'bird';
  csTinyImageLabel[3]:= 'cat';
  csTinyImageLabel[4]:= 'deer';
  csTinyImageLabel[5]:= 'dog';
  csTinyImageLabel[6]:= 'frog';
  csTinyImageLabel[7]:= 'horse';
  csTinyImageLabel[8]:= 'ship';
  csTinyImageLabel[9]:= 'truck';
end;  

 const TBATCH= 'C:\Program Files\Streaming\maxbox4\maxbox47590\maxbox4\cifar-10-batches-bin\test_batch.bin';

procedure TestBatchPerformance(aNN: TNNet; Testpath: string; aimgvolumes:TNNetVolumeList);
var Rate,Loss,ErrorSum,LastError: TNeuralFloat;
    ImgVolumes: TNNetVolumeList;
begin
    rate:=0; loss:=0;
    ErrorSum:=0;
    ImgVolumes:= TNNetVolumeList.Create(true);
    try 
    // creates required volumes to store images
      for it:= 0 to 9999 do 
        ImgVolumes.Add(TNNetVolume.Create());
      loadCifar10Dataset(ImgVolumes, TBATCH, 0,csEncodeRGB); //"test_batch.bin"  }
      writeln('testpath '+testpath)
      //if pos('Resize40',Testpath) > 0 then
         ImgVolumes.ResizeImage(64, 64); 
      WriteLn(' Testbatch size: '+ itoa(ImgVolumes.gettotalsize));
      WriteLn(' dbug volumescount: '+ itoa(ImgVolumes.Count));
      //ImgVolumes[it].ReSize(40,40,3);
      TestBatch(aNN, ImgVolumes, 500, rate,loss,ErrorSum);
      writeln('Testbatch score: '+Format('Rate:%.4f, Loss:%.4f, ErrorSum:%.4f',
                                                      [rate, loss, ErrorSum]));
      label2.Caption:= format('Score Ø %.2f%% ',[rate*100]);
    finally
      ImgVolumes.Free;
  end;  
end;     


//{$R *.lfm}

procedure loadSuperForm;
begin
Form1:= TForm1.create(self)
with Form1 do begin
  Left := 587
  Height:= 491
  Top:= 249
  Width:= 958
  BorderStyle:= bsSingle; //bsDialog
  Caption:= 'CIFAR-10 Super Resolution App maXbox4'
  ClientHeight:= 491
  ClientWidth:= 958
  icon.loadfromresourcename(hinstance,'ZCUBE');
  //DesignTimePPI:= 120
  OnClose:= @TForm1FormClose;
  OnCreate:= @TForm1FormCreate;
  //LCLVersion:= '2.0.2.0'
  Image1:= TImage.create(form1)
  Show;
  with image1 do begin
    parent:= form1
    Left:= 40
    Height:= 32
    Top:= 50
    Width:= 32
    Stretch:= True
  end;
  Label1:= TLabel.create(form1)
  with label1 do begin
   parent:= form1
    Left:= 580
    Height:= 20
    Top:= 344
    Width:= 73
    Caption:= 'Label Class'
    font.size:= 14;
    ParentColor:= False
    ParentFont:= False
  end ;
  ImageGray:= TImage.create(form1)
  with imagegray do begin
   parent:= form1;
    Left:= 220
    Height:= 64
    Top:= 50
    Width:= 64
    Stretch:= True
  end;
  //&object ImageHE: TImage
  Imagehe:= TImage.create(form1)
  with imagehe do begin
   parent:= form1;
    Left:= 400
    Height:= 128
    Top:= 50
    Width:= 128
    Stretch:= True
  end;
  Imageve:= TImage.create(form1)
  with imageve do begin
   parent:= form1;
    Left:= 580
    Height:= 256
    Top:= 50
    Width:= 256
    Stretch:= True
  end ;
   BitBtn13:=  TBitBtn.create(form1)
   with bitbtn13 do begin
    parent:= form1
    Left:= 40
    Height:= 45
    Top:= 344
    Width:= 182
    Caption:= ' RUN SUPER'
    OnClick:= @Tform1BitBtn13Click;
    glyph.loadfromresourcename(hinstance,'WET2');
    ParentFont:= False
    TabOrder:= 0
  end ;
  Label2:= TLabel.create(form1)
  with label2 do begin
   parent:= form1
    Left:= 40
    Height:= 20
    Top:= 25
    Width:= 96
    Caption:= 'Original 32x32'
    ParentColor:= False
    ParentFont:= False
  end;
  Label3:= TLabel.create(form1)
  with label3 do begin
   parent:= form1
    Left:= 220
    Height:= 20
    Top:= 25
    Width:= 113
    Caption:= 'Computed 64x64'
    ParentColor:= False
    ParentFont:= False
  end ;
  Label4:= TLabel.create(form1)
  with label4 do begin
   parent:= form1
    Left:= 400
    Height:= 20
    Top:= 25
    Width:= 129
    Caption:= 'Computed 128x128'
    ParentColor:= False
    ParentFont:= False
  end ;
  Label5:= TLabel.create(form1)
  with label5 do begin
   parent:= form1
    Left:= 580
    Height:= 20
    Top:= 25
    Width:= 129
    Caption:= 'Computed 256x256'
    ParentColor:= False
    ParentFont:= False
  end;
   CheckPause:= TCheckBox.create(form1)
   with checkpause do begin
    parent:= form1
    Left:= 40
    Height:= 24
    Top:= 400
    Width:= 191
    Caption:= 'Run Pause'
    ParentFont:= False
    TabOrder:= 1
  end ;
 end;
 TForm1FormCreate(self);
end; 

{ TForm1 }
procedure TForm1BitBtn13Click(Sender: TObject);
var
  NN2, NN3, NN41: THistoricalNets;
  Img: TTinyImage;
  //cifarFile: TTInyImageFile;
  I, K: integer;
  ImgVolumes: TNNetVolumeList;
  Volume: TNNetVolume;
  vInput, vOutput, vBigOutput, vDisplay: TNNetVolume;
begin
  if not(CheckBinFiles()) then begin
    writeln('CheckBinFiles not OK');
    //exit;
  end;

  Randomize;
  BitBtn13.Enabled := false;
  writeln('Creating Neural Network...');
  ImgVolumes:= TNNetVolumeList.Create(true);

  try
     NN2 := CreateResizingNN(32, 32, csExampleFileName);
     NN2.DebugStructure();
     writeln('after debug struct');
     //TestBatchPerformance(NN2, csExampleFileName);             
     NN3 := CreateResizingNN(64, 64, csExampleFileName);
     sleep(400)
     //NN41 := CreateResizingNN(128, 128, csExampleFileName);

     writeln('Loading Tiny Cifar Images...');
     for I:= 0 to 9999 do begin
       Volume := TNNetVolume.Create();
       ImgVolumes.Add(Volume);
     end;

   loadCifar10Dataset2(ImgVolumes, 1, 0,csEncodeRGB);   //1 is data_batch_1.bin
   //AssignFile(cifarFile, 'data_batch_1.bin');
   //Reset(cifarFile);
   I := 1;
   writeln('image count and testbatch: '+itoa(ImgVolumes.count ));
  // TestBatchPerformance(NN3, csExampleFileName, imgvolumes);  
  //while not EOF(cifarFile) do
  for it:= 1 to 200 do begin
    //Read(cifarFile, Img);
    LoadNNetVolumeIntoTinyImage4(imgvolumes[it], img);
    Label1.Caption := 'label: '+csTinyImageLabel[Img.bLabel];
    writeln('debug '+csTinyImageLabel[Img.bLabel]);
    Volume := TNNetVolume.Create();
    writeln('debug1 '+csTinyImageLabel[Img.bLabel]);
    LoadTinyImageIntoNNetVolume1(Img, Volume);
    //imgvolumes[it].divi75(64);
    Volume.Divi75(64);
    Volume.Sub19(2);
    //imgvolumes[it].Sub64(2);
    //writeln('debug3 '+csTinyImageLabel[Img.bLabel]);
    ImgVolumes.Add(Volume);
    
    if (I mod 100 = 0) then begin
      LoadTinyImageIntoTImage(Img, Image1);
      LoadTinyImageIntoTImage(Img, ImageGray);
      LoadTinyImageIntoTImage(Img, ImageHE);
      LoadTinyImageIntoTImage(Img, ImageVE);
      Image1.Width := 128;
      Image1.Height := 128;
      ImageGray.Width := 128;
      ImageGray.Height := 128;
      ImageHE.Width := 128;
      ImageHE.Height := 128;
      ImageVE.Width := 256;
      ImageVE.Height := 256;
      Application.ProcessMessages;
    end;
    inc(I);
  end;
  //CloseFile(cifarFile);

  vInput     := TNNetVolume.Create0(32,32,3,0);
  vOutput    := TNNetVolume.Create0(64,64,3,0);
  vDisplay   := TNNetVolume.Create();
  vBigOutput := TNNetVolume.Create0(64,64,3,0);

  WriteLn('128x128 -> 256x256 Neural network has: ');
  WriteLn(' Layers: '+itoa( NN3.CountLayers()  ));
  WriteLn(' Neurons:'+itoa(NN3.CountNeurons() ));
  WriteLn(' Weights:'+itoa(NN3.CountWeights() ));
  WriteLn('Computing SuperResolution...');

  while not (FWantQuit) do begin
    I := random(1000);
    vInput.Copy(ImgVolumes[I]);
    Label1.Caption:= 'label: '+csTinyImageLabel[ ImgVolumes[I].Tag ];
    vDisplay.Copy(vInput);
    vDisplay.Add13(2);
    vDisplay.Mul26(64);
    LoadNNetVolumeIntoTinyImage4(vDisplay, Img);
    //writeln('debug4 '+csTinyImageLabel[Img.bLabel]);
    LoadTinyImageIntoTImage(Img, Image1);
    
    NN2.Compute65(vInput,0); NN2.GetOutput(vBigOutput);
    vDisplay.Copy(vBigOutput);
    vDisplay.Add13(2);
    vDisplay.Mul26(64);
    //LoadVolumeIntoTImage(vDisplay, ImageVE, csEncodeRGB);
    LoadVolumeIntoImage(vDisplay, ImageGray);
    ImageGray.Width := 128;
    ImageGray.Height := 128;

    NN3.Compute65(vBigOutput,0);  NN3.GetOutput(vBigOutput);
    vDisplay.Copy(vBigOutput);
    vDisplay.Add13(2);
    vDisplay.Mul26(64);
    LoadVolumeIntoImage(vDisplay, ImageHE);
    ImageHE.Width := 128;
    ImageHE.Height := 128;

    {NN41.Compute65(vBigOutput,0);  NN41.GetOutput(vBigOutput);
    vDisplay.Copy(vBigOutput);
    vDisplay.Add13(2);
    vDisplay.Mul26(64);
    LoadVolumeIntoImage(vDisplay, ImageVE);
    ImageVE.Width := 256;
    ImageVE.Height := 256;  //}

    for K := 1 to 50 do begin                                 
      Application.ProcessMessages();
      Sleep(100);
    end;
    while CheckPause.Checked do begin
      Application.ProcessMessages();
      Sleep(100);
      FWantQuit := true; 
      CheckPause.Caption:= 'Run Pause or Close'
    end;
  end; //while
  finally
     NN2.Free;
     NN3.Free;
     //NN41.Free;
     vBigOutput.Free;
     vDisplay.Free;
     vInput.Free;
     vOutput.Free;                             
     ImgVolumes.Free;
     BitBtn13.Enabled := true;
     FWantQuit := false; 
     writeln('debug finally and frees..');
  end;   
end;

procedure TForm1FormClose(Sender: TObject; var CloseAction: TCloseAction);
begin
  FWantQuit := true;
    {NN2.Free;
     NN3.Free;
     //NN4.Free;
     vBigOutput.Free;
     vDisplay.Free;
     vInput.Free;
     vOutput.Free;                             
     ImgVolumes.Free;   }
  writeln('form close...');
end;

procedure TForm1FormCreate(Sender: TObject);
begin
  FWantQuit := false;
  writeln('form createp sueperres Iü');
end;

function CheckBinFiles: boolean;
begin
  Result := true;
  if not (FileExists('data_batch_1.bin')) then begin
    Result := false;
    ShowMessage('CIFAR-10 files have not been found.' + Chr(13) +
      'Please download from https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz');
  end;
end;

begin //@main

  setClassifierLabelsTiny;
  loadsuperForm();

 end.
End.

ref: https://github.com/maxkleiner/neural-api/blob/master/examples/SuperResolution/usuperesolutionapp.pas

