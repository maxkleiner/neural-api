unit Uvisualcifar10test_mX4_14;

//{$mode objfpc}{$H+}
//https://sourceforge.net/p/cai/svncode/HEAD/tree/trunk/lazarus/experiments/visualCifar10test/uvisualcifar10test.lfm
//https://maxbox4.wordpress.com/2022/07/07/cnn-pipeline-train/  
{get pretrained *.nn model at: https://github.com/maxkleiner/neural-api/tree/master/examples/SimpleImageClassifier}      

interface
                                                                    
{uses
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls,
  ExtCtrls, neuralnetwork, neuraldatasetsv, neuralvolumev, neuraldatasets,
  neuralvolume; }
  
Const //PReModel_NN = 'SimpleSeparableImageClassifier124_50_21.nn'; 
      //PReModel_NN = 'SimpleSeparableImageClassifier.nn';  
      //PReModel_NN = 'SimpleSeparableImageClassifier124.nn';
      //PReModel_NN = 'SimpleSeparableImageClassifier124_50_3.nn';
      //PReModel_NN = 'ImageClassifierSELU_Tutor89.nn';
      //PReModel_NN = 'EKON25_SimpleImageClassifier-60.nn';
      PReModel_NN = '..\examples\SimpleImageClassifierEkon26_70.nn';
      //PReModel_NN = '1076_SimpleImageClassifierEKON25_5000.nn';
      //PReModel_NN = 'C:\maXbox\works2021\maxbox4\crypt\yolo-tiny.h5';
type

  { TFormVisualLearning }

  TFormVisualLearning = {class(}TForm;
  var
    ButTest, btnpredict: TButton;
    EdTestBinFile: TEdit;
    ImgSample: TImage;
    LabClassRate: TLabel;
    LabTestFile, labclasslabel, lblchoice, lbl5, lbl51, lbl6, lbl7: TLabel;
    lbl1, lbl2, lbl3, lbl31, lbl4, lbl41, lbl8, lbl9, lbl10: TLabel;
    OpenDialogNN: TOpenDialog;
    ImgSample1, imgsample2, imgsample3, imgsample4, imgsample5, 
               img7, img8, img9, img10, img6: TImage;
    procedure TFormVisualLearningButTestClick(Sender: TObject);
    procedure TFormVisualLearningBtnPredictClick(Sender: TObject);
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
  FrmVisualLearn: TFormVisualLearning;

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
  end else begin
    //if (OpenDialogNN.Execute()) then begin
    OpenDialogNN.filename := Exepath+PReModel_NN; 
      labtestfile.Caption:= 'CIFAR-10 testing file: '+Exepath+PReModel_NN;
      if FileExists(OpenDialogNN.FileName) then begin
        FRunning := true;
        ButTest.Caption := 'Stop CNN';
        TFormVisualLearningLearn(Sender);
        FRunning := false;
        ButTest.Caption := 'Retest CNN';
      end;
    end;
  //end;
end;

procedure TFormVisualLearningBtnPredictClick(Sender: TObject);
begin
  if not CheckCIFARFile() then writeln('___NO cifar datasets found___'); //exit;
  if (FRunning) then begin
    FRunning := false;
    writeln('not implementd yet')
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

var  cs10Labels: array[0..9] of string;
procedure setClassifierLabels;
begin
  cs10Labels[0]:= 'airplane';
  cs10Labels[1]:= 'automobile';
  cs10Labels[2]:= 'bird';
  cs10Labels[3]:= 'cat';
  cs10Labels[4]:= 'deer';
  cs10Labels[5]:= 'dog';
  cs10Labels[6]:= 'frog';
  cs10Labels[7]:= 'horse';
  cs10Labels[8]:= 'ship';
  cs10Labels[9]:= 'truck';
end;  

//var NN: TNNet; //just to test the predictor

function TFormVisualPredictor(ImgIdx: integer): string;
var
  NN: TNNet; NeuralFit: TNeuralImageFit;
  ImgVolumes: TNNetVolumeList;
  Volume: TNNetVolume;
  pOutput, vOutput, vDisplay: TNNetVolume;
  hit, miss, NumClasses, I: integer;
  ErrorSum, LastError: TNeuralFloat;
  startTime, totalTimeSeconds: double;
  fileName: string;
  Rate, Loss: TNeuralFloat;
 
begin
  writeln('Creating CNeural Network...');
  ImgVolumes := TNNetVolumeList.Create(true);
  NumClasses := 10;
  fileName:= Exepath+PReModel_NN; //OpenDialogNN.FileName;     nn.60?

  //--------------------------------------------------------------------
  // creates required volumes to store images
  for I:= 0 to 9999 do begin
    Volume := TNNetVolume.Create();
    ImgVolumes.Add(Volume);
  end;
  //--------------------------------------------------------------------

  NN:= TNNet.Create();
  writeln('Loading predict neural network from file: '+fileName);
  NN.LoadFromFile(fileName);
  NN.EnableDropouts(false);
  pOutput := TNNetVolume.Create0(NumClasses,1,1,0);   //or 1
  //NN.DebugWeights();
  //WriteLn(' Layers: ',  NN.CountLayers()  );
  WriteLn('Loaded Neural CNN network has: ');
  WriteLn(' Layers: '+ itoa(NN.CountLayers()  ));
  WriteLn(' Neurons: '+ itoa(NN.CountNeurons() ));
  WriteLn(' Weights: '+ itoa(NN.CountWeights() ));
  WriteLn('CNN Predictor Computing...');
  try
    startTime:= Now();
    loadCifar10Dataset2(ImgVolumes, 5, 0,csEncodeRGB);   //eval data_batch_5.bin" 5
    WriteLn(' Totalsize: '+ itoa(ImgVolumes.gettotalsize));
    WriteLn(' dbug volumescount: '+ itoa(ImgVolumes.Count));
    
     //ImgIdx := Random(ImgVolumes.Count);
     //-- CAREFUL with that axe again
     //procedure Compute(pInput: TNNetVolume; FromLayerIdx:integer= 0); overload;
     //neuralfit.ClassifyImage(NN, ImgVolumes[ImgIdx],  pOutput);
     NN.Compute65(ImgVolumes[ImgIdx],0);     //fit and predict
     NN.GetOutput(pOutput);
     result:= cs10labels[pOutput.GetClass()];
  finally 
    NN.Free;
    pOutput.Free;
    Volume.Free;
    ImgVolumes.Free;
  end;  
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
  NeuronCnt: integer;
  ImgIdx, simgdx, simginc: integer;
  MaxW, MinW: TNeuralFloat;
  fileName: string;
  firstNeuroLayer: integer;
  Rate, Loss: TNeuralFloat;
 
begin
  writeln('Creating CNeural Network...');
  ImgVolumes := TNNetVolumeList.Create(true);
  NumClasses := 10;
  //fileName:= {Exepath+}PReModel_NN; //OpenDialogNN.FileName;
  fileName:= Exepath+PReModel_NN; //OpenDialogNN.FileName;   nn.60?

  //--------------------------------------------------------------------
  // creates required volumes to store images
  for I:= 0 to 9999 do begin
    Volume := TNNetVolume.Create();
    ImgVolumes.Add(Volume);
  end;
  //--------------------------------------------------------------------

  NN:= TNNet.Create();
  writeln('Loading neural network from file: '+fileName);
  NN.LoadFromFile(fileName);
  NN.EnableDropouts(false);
  firstNeuroLayer := NN.GetFirstNeuronalLayerIdx(0);
  pOutput := TNNetVolume.Create0(NumClasses,1,1,0);   //or 1
  vOutput := TNNetVolume.Create0(NumClasses,1,1,0);
  vDisplay:= TNNetVolume.Create0(NumClasses,1,1,0);
  SetLength(aImage, NN.Layers[firstNeuroLayer].Neurons.Count);
  SetLength(sImage, NumClasses);

  for NeuronCnt:= 0 to NN.Layers[firstNeuroLayer].Neurons.Count-1 do begin
    aImage[NeuronCnt]:= TImage.Create(FrmVisualLearn);
    aImage[NeuronCnt].Parent := FrmVisualLearn;
    aImage[NeuronCnt].Width  := 
              NN.Layers[firstNeuroLayer].Neurons[NeuronCnt].Weights.SizeX;
    aImage[NeuronCnt].Height := 
              NN.Layers[firstNeuroLayer].Neurons[NeuronCnt].Weights.SizeY;
    aImage[NeuronCnt].Top    := (NeuronCnt div 12) * 38+ 160;   //120
    aImage[NeuronCnt].Left   := (NeuronCnt mod 12) * 38+ 60;
    aImage[NeuronCnt].Stretch:=true;
  end;
  for simgdx:= 0 to NumClasses- 1 do begin
    sImage[simgdx]:= TImage.Create(FrmVisualLearn);
    sImage[simgdx].Parent := FrmVisualLearn;
    sImage[simgdx].Width  := 32;
    sImage[simgdx].Height := 32
    sImage[simgdx].Top    := (simgdx div 12) * 38+ 460;   //120
    sImage[simgdx].Left   := (simgdx mod 12) * 38+ 60;
    //aImage[simgdx].Stretch:=true;
  end;

  NN.DebugWeights();
  //WriteLn(' Layers: ',  NN.CountLayers()  );
  WriteLn('Loaded Neural CNN network has: ');
  WriteLn(' Layers: '+ itoa(NN.CountLayers()  ));
  WriteLn(' Neurons: '+ itoa(NN.CountNeurons() ));
  WriteLn(' Weights: '+ itoa(NN.CountWeights() ));
  WriteLn('CNN Computing...');
  begin
    hit:= 0; miss := 0;
    ErrorSum:= 0; LastError := 0;
    startTime:= Now();
    simginc:= 0;
    //loadCifar10Dataset(ImgVolumes, EdTestBinFile.Text);  //"data_batch_5.bin"
    loadCifar10Dataset2(ImgVolumes, 5, 0,csEncodeRGB);   //test batch file as 5
    WriteLn(' Totalsize: '+ itoa(ImgVolumes.gettotalsize));
    WriteLn(' dbug volumescount: '+ itoa(ImgVolumes.Count));
    for I := 0 to ImgVolumes.Count - 1 do begin
      if not(FRunning) then Break;
      ImgIdx := Random(ImgVolumes.Count);
      //writeln('img idx: '+itoa(imgidx));
      //labclasslabel.caption:=  cs10Labels[(ImgVolumes[ImgIdx].Tag)];
      //-- CAREFUL
      //procedure Compute(pInput: TNNetVolume; FromLayerIdx:integer = 0); overload;
      NN.Compute65(ImgVolumes[ImgIdx],0);     //fit and predict
      NN.GetOutput(pOutput);
      vOutput.SetClassForReLU( ImgVolumes[ImgIdx].Tag ); // ReLU - no softmax
      ErrorSum:= ErrorSum +vOutput.SumDiff(pOutput);

      if I mod 1000 = 0 then begin
        vDisplay.Copy(ImgVolumes[ImgIdx]);
        vDisplay.Mul26(64);
        vDisplay.Add13(128);
        labclasslabel.caption:= cs10Labels[(ImgVolumes[ImgIdx].Tag)];
        LoadVolumeIntoTImage(vDisplay, ImgSample, csEncodeRGB);
        ImgSample.Width := 64+40;
        ImgSample.Height := 64+40;
        //ImgSample.Stretch:= True;
        //sImage[I].assign(imgsample);
        sImage[simginc].picture:= imgsample.picture;
        inc(simginc);

        for NeuronCnt:= 0 to NN.Layers[firstNeuroLayer].Neurons.Count-1 do begin
          MaxW:= NN.Layers[firstNeurolayer].Neurons[NeuronCnt].Weights.GetMax();
          MinW:= NN.Layers[firstNeuroLayer].Neurons[NeuronCnt].Weights.GetMin();
          vDisplay.Copy(NN.Layers[firstNeuroLayer].Neurons[NeuronCnt].Weights);
          vDisplay.Mul26(256/(MaxW-MinW));
          vDisplay.Add13(128);
          LoadVolumeIntoTImage(vDisplay, aImage[NeuronCnt],csEncodeRGB);
          aImage[NeuronCnt].Width := 35;
          aImage[NeuronCnt].Height:= 35;
        end;
        Application.ProcessMessages();
      end;
      Application.ProcessMessages();

      if pOutput.GetClass() = ImgVolumes[ImgIdx].Tag then begin
        Inc(Hit);
        //Writeln('dbug '+itoa(pOutput.GetClass()));
      end else begin
        Inc(Miss);
      end;

      if (Hit>0) and (I>0) and ((I+1) mod 1000 = 0) then begin
        totalTimeSeconds:= (Now() - startTime) * 24 * 60 * 60;
        WriteLn
        ( { I+1, ' Accuracy:'+ Hit/(Hit+Miss):6:4,
          ' Error:', (ErrorSum-LastError):10:5,
          ' Time:', totalTimeSeconds:6:2,'s',
          ' Forward:', (NN.ForwardTime * 24 * 60 * 60):6:2,'s',
          ' Backward:', (NN.BackwardTime * 24 * 60 * 60):6:2,'s' }
          itoa(I+1)+ ' Accuracy:'+ flots(Hit/(Hit+Miss))+
          ' Error:'+flots((ErrorSum-LastError))+
          ' Time:'+flots(totalTimeSeconds)+'s'+
          ' Forward:'+ flots(NN.ForwardTime * 24 * 60 * 60)+'s'+
          ' Backward:'+flots(NN.BackwardTime* 24 * 60 * 60)+'s'  
        );
        NN.ClearTime();
        LabClassRate.Caption:= format('%.2f%% ',[(Hit*100)/(Hit+Miss)]);
        startTime:= Now();
        LastError:= ErrorSum;
        Application.ProcessMessages;
      end;
    end;

    NN.DebugWeights();
   // This function tests a neural network on the passed ImgVolumes
  {procedure TestBatch
  ( NN: TNNet; ImgVolumes: TNNetVolumeList; SampleSize: integer;
   out Rate, Loss, ErrorSum: TNeuralFloat
   ); }
    rate:= 0; loss:= 0;
    ErrorSum:= 0;
    TestBatch(NN, ImgVolumes, 1000, rate,loss,ErrorSum);
    writeln('End Test batch score: '+Format(' Rate:%.4f, Loss:%.4f, ErrorSum:%.4f ',
                                          [rate, loss, ErrorSum]));
    LabClassRate.Caption:= format('Ø %.2f%% ',[rate*100]);
  end;
  for NeuronCnt := Low(aImage) to High(aImage) do 
    aImage[NeuronCnt].Free;
  for it:= Low(sImage) to High(sImage) do begin
    //sImage[it].Free;
  end;
  NN.Free;
  vDisplay.Free;
  vOutput.Free;
  pOutput.Free;
  ImgVolumes.Free;
end;

procedure loadsetPicture(ImgVol: TNNetVolumeList; idx:integer; 
                           vdisplay: TNNetVolume; img:TImage; lbl:Tlabel);
var color_enc,imsize: integer;
begin
  color_enc:= csEncodeRGB;
  imsize:= 64;
  vDisplay.Copy(ImgVol[idx]);
  vDisplay.NeuronalInputToRgbImg(color_enc);
  LoadVolumeIntoTImage(vDisplay, Img, color_enc);
  Img.Width:=imsize; Img.Height:=imsize;      //}
  lbl.caption:= cs10labels[(ImgVol[idx].Tag)]
end;    

procedure {TFormVisualLearning.}LoadPicture2(Sender: TObject);
var
  aLabelX, aLabelY: array of TLabel;
  vDisplay: TNNetVolume;  
  fileNameBase, fileNameImage, fileNameNeuroImage: string;
  fileNameConfusion: string;
  CSVFile, CSVConfusion: TextFile;
   color_encoding, imsize: integer;
  iInputDepth: integer;
  InputLayer: TNNetLayer;
  AccuracyWithInertia: TNeuralFloat;
  ValidationRecord: TNeuralFloat;
   FIBDoc: TIBDataSet;
   csMachineAnimalCifar10Labels: array[0..9] of string; //unknown so far fix
   ownobj: boolean;
  ImgVolumes, ImgTestVolumes, ImgValidationVolumes: TNNetVolumeList;  
  globalImgInput: TNNetVolume;
  {   csMachineAnimalCifar10Labels: array[0..9] of string =
  ( 'airplane',
    'automobile',
    'ship',  // used to be bird
    'truck', // used to be cat
    'deer',  // used to be deer
    'dog',   // used to be dog
    'frog',  // used to be frog
    'horse', // used to be horse
    'bird',  // used to be ship
    'cat'    // used to be truck
  );  }
  
begin
  ValidationRecord:= 0.0;
  AccuracyWithInertia:= 10;
  imsize:= 64;
  color_encoding:= 1; //ComboColor.ItemIndex;
  if color_encoding = csEncodeGray then begin
    iInputDepth:= 1;
  end else 
    iInputDepth:= 3;

  {iAlgo := ComboAlgo.ItemIndex;
  iInputPadding := StrToInt(EdInputPadding.Text);
  iPadding := StrToInt(EdConvPadding.Text);
  iInnerConvFeatureSize := StrToInt(EdInnerConvFeatureSize.Text);    }
  
  fileNameImage:= fileNameBase + '.bmp';
  fileNameNeuroImage:= fileNameBase + '-neurons.bmp';
  fileNameConfusion:= fileNameBase + '-confusion.csv';
  //AssignFile(CSVFile, FileNameCSV);
  //AssignFile(CSVConfusion, fileNameConfusion);
  //LoadCifar10binFilesIfRequired();
  
  if not CheckCIFARFile() then begin
      //application.Terminate;
      writeln('cifar imagefiles not found..')
    end;
  try  
  //LoadCifar10binFilesIfRequired();  
  color_encoding:= csEncodeRGB;
  ownobj:= True;
  ImgVolumes:= TNNetVolumeList.Create(ownobj);   //train block
  ImgVolumes.FreeObjects:= ownobj;                // no effect!
  writeln('train block count: '+itoa(imgvolumes.count));       
  ImgTestVolumes:= TNNetVolumeList.Create(ownobj);
  ImgValidationVolumes:= TNNetVolumeList.Create(ownobj);     //own object!
  globalImgInput:= TNNetVolume.Create();  
  
  CreateCifar10Volumes(ImgVolumes, ImgValidationVolumes, 
                           ImgTestVolumes, csEncodeRGB);
  writeln('test block count2: '+itoa(ImgTestVolumes.count));                          
                           
  {LoadCifar10binFilesIfRequired();  
  loadCifar10Dataset6(ImgVolumes, 1, 0, color_encoding);
  loadCifar10Dataset6(ImgVolumes, 2, 10000, color_encoding);
  loadCifar10Dataset6(ImgVolumes, 3, 20000, color_encoding);
  loadCifar10Dataset6(ImgVolumes, 4, 30000, color_encoding);
  loadCifar10Dataset6(ImgValidationVolumes, 5, 0, color_encoding);  //((}
  //loadCifar10Dataset(ImgTestVolumes, 'test_batch.bin', 0, color_encoding);
  //TranslateCifar10VolumesToMachineAnimal(ImgVolumes);              //fix
  //TranslateCifar10VolumesToMachineAnimal(ImgValidationVolumes);
  //TranslateCifar10VolumesToMachineAnimal(ImgTestVolumes);

  WriteLn('Training Images:'+itoa(ImgVolumes.Count) +' Test Images:'+ itoa(ImgTestVolumes.Count)+ ' Validation Images:'+itoa(ImgValidationVolumes.Count));

  //FFinishedThread.ReSize(FThreadNum);
  //FThreadNN := TNNetDataParallelism.Create74(NN, FThreadNum, false);
  //--------------------------------------------------------------------
  vDisplay:= TNNetVolume.Create0(10,1,1,0);
  
  loadsetPicture(ImgTestVolumes,0, vdisplay,ImgSample1, lbl1);

  loadsetPicture(ImgTestVolumes,1, vdisplay,ImgSample2, lbl2);
    
  loadsetPicture(ImgValidationVolumes,1233, vdisplay,ImgSample3, lbl3);    
    writeln('predict: '+TFormVisualPredictor(1233));
    writeln('real: '+lbl3.caption);
    lbl31.caption:= 'predict: '+TFormVisualPredictor(1233);
  
  loadsetPicture(ImgValidationVolumes,333, vdisplay,ImgSample4, lblchoice);
   writeln('predict: '+TFormVisualPredictor(333));
    writeln('real: '+lblchoice.caption);
    lbl41.caption:= 'predict: '+TFormVisualPredictor(333);
   
  loadsetPicture(ImgValidationVolumes,430, vdisplay,ImgSample5, lbl5);   
    writeln('predict: '+TFormVisualPredictor(430));
    writeln('real: '+lbl5.caption);
    lbl51.caption:= 'predict: '+TFormVisualPredictor(430);
   
  loadsetPicture(ImgVolumes,430, vdisplay,Img6, lbl6); 
    
  loadsetPicture(ImgVolumes,6, vdisplay,Img7, lbl7); 
    
  loadsetPicture(ImgVolumes,7, vdisplay,Img8, lbl8); 
       
  loadsetPicture(ImgVolumes,8, vdisplay,Img9, lbl9); 
    
  loadsetPicture(ImgVolumes,9, vdisplay,Img10, lbl10);       
       
   (* SetLength(aImage,  33);
    SetLength(aLabelY, 100{RowNum});
    SetLength(aLabelX, 100 {ColNum});
   CreateAscentImages
  ( nil {GrBoxNeurons},
    aImage, aLabelX, aLabelY, {ImageCount=}32,
    {InputSize=}64, {displaySize=}128, {ImagesPerRow=}8
  ); *)
   
 finally
  globalImgInput.Free;
  //ImgValidationVolumes.Clear;
  ImgVolumes.FreeObjects:= ownobj; 
  writeln('state of owner '+botostr(ImgVolumes.FreeObjects));
  ImgValidationVolumes.Free;
  ImgTestVolumes.Free;
  ImgVolumes.Free;
  //CloseFile(CSVFile);
  //CloseFile(CSVConfusion);
  vDisplay.Free;
  writeln('free picture ressources..')
 end; 
end;


procedure loadCifarForm;
begin
FrmVisualLearn:= TFormVisualLearning.create(self)
with FrmVisualLearn do begin
  setbounds(2664,194,964,616)
  Caption:= 'Visual CIFAR-10 pretrained NN Testing maXbox4 2023'
  ClientHeight:= 616; ClientWidth:= 964;
  //DesignTimePPI:= 120
  formstyle:= fsstayontop;
  OnClose:= @TFormVisualLearningFormClose;
  OnCreate:= @TFormVisualLearningFormCreate;
  Position:= poScreenCenter
  icon.loadfromresourcename(hinstance,'XDIRECTX');
  //LCLVersion:= '2.0.2.0'
  Show();
  ButTest:= TButton.create(self)
  with buttest do begin
   parent:= FrmVisualLearn;
   setbounds(350,68,200,35)
    Caption:= '&Test Neural Network'
    OnClick:= @TFormVisualLearningButTestClick
    ParentFont:= False
    TabOrder:= 0
  end;
  Btnpredict:= TButton.create(self)
  with btnpredict do begin
   parent:= FrmVisualLearn;
   setbounds(350,110,200,35)
    Caption:= '&Predict Neural Network...'
    OnClick:= @TFormVisualLearningbtnpredictClick
    ParentFont:= False
    TabOrder:= 0
  end;
  ImgSample:= TImage.create(self)
  with imgsample do begin
   parent:= FrmVisualLearn;
   setbounds(15,15,34,33)
   Stretch:= True
  end;
  LabTestFile:= TLabel.create(self)
  with labtestfile do begin
   parent:= FrmVisualLearn;
    setbounds(350,10,137,20)
    Caption:= 'CIFAR-10 testing file:'
    ParentColor:= False
    ParentFont:= False
  end;
  Labclasslabel:= TLabel.create(self)
  with labclasslabel do begin
   parent:= FrmVisualLearn;
    setbounds(15,125,137,20)
    Caption:= 'CIFAR-10 label:'
    ParentColor:= False
    ParentFont:= False
  end;
  EdTestBinFile:= TEdit.create(self)
  with EdTestBinFile do begin
   parent:= FrmVisualLearn;
    setbounds(350,32,300,28)
    ParentFont:= False
    Font.size:= 12;
    TabOrder:= 1
    Text:= 'test_batch.bin'
  end;
  LabClassRate:= TLabel.create(self)
  with labclassrate do begin
   parent:= FrmVisualLearn;
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
  ImgSample1:= TImage.create(FrmVisualLearn)
  with imgsample1 do begin
   parent:= FrmVisualLearn;
   setbounds( 700, 32, 32, 32)
    Stretch := True
    lbl1:= Tlabel.create(FrmVisualLearn)
    lbl1.parent:=FrmVisualLearn;
    lbl1.setbounds(700, 96, 62, 32);
  end;
  ImgSample2:= TImage.create(FrmVisualLearn)
  with imgsample2 do begin
   parent:= FrmVisualLearn;
   setbounds( 800, 32, 32, 32)
    Stretch := True
    lbl2:= Tlabel.create(FrmVisualLearn)
    lbl2.parent:=FrmVisualLearn;
    lbl2.setbounds(800, 96, 62, 32);
  end;
  ImgSample3:= TImage.create(FrmVisualLearn)
  with imgsample3 do begin
   parent:= FrmVisualLearn;
   setbounds( 700, 132, 32, 32)
    Stretch := True
    lbl3:= Tlabel.create(FrmVisualLearn)
    lbl3.parent:=FrmVisualLearn;
    lbl3.setbounds(700, 196, 62, 32);
    lbl31:= Tlabel.create(FrmVisualLearn)
    lbl31.parent:=FrmVisualLearn;
    lbl31.setbounds(700, 210, 62, 32);
  end;
  ImgSample4:= TImage.create(FrmVisualLearn)
  with imgsample4 do begin
   parent:= FrmVisualLearn;
   setbounds( 800, 132, 32, 32)
    Stretch := True
    lblchoice:= Tlabel.create(FrmVisualLearn)
    lblchoice.parent:=FrmVisualLearn;
    lblchoice.setbounds(800, 196, 62, 32);
    lblchoice.caption:= 'this a test class';
    lbl41:= Tlabel.create(FrmVisualLearn)
    lbl41.parent:=FrmVisualLearn;
    lbl41.setbounds(800, 210, 62, 32);
  end;
  ImgSample5:= TImage.create(FrmVisualLearn)
  with imgsample5 do begin
   parent:= FrmVisualLearn;
   setbounds( 700, 232, 32, 32)
    Stretch := True
    lbl5:= Tlabel.create(FrmVisualLearn)
    lbl5.parent:=FrmVisualLearn;
    lbl5.setbounds(700, 296, 62, 32);
    lbl51:= Tlabel.create(FrmVisualLearn)
    lbl51.parent:=FrmVisualLearn;
    lbl51.setbounds(700, 310, 62, 32);
  end;
  Img6:= TImage.create(FrmVisualLearn)
  with img6 do begin
   parent:= FrmVisualLearn;
   setbounds( 700, 332, 32, 32)
    Stretch := True
    lbl6:= Tlabel.create(FrmVisualLearn)
    lbl6.parent:=FrmVisualLearn;
    lbl6.setbounds(700, 396, 62, 32);
  end;
  Img7:= TImage.create(FrmVisualLearn)
  with img7 do begin
   parent:= FrmVisualLearn;
   setbounds( 800, 332, 32, 32)
    Stretch := True
    lbl7:= Tlabel.create(FrmVisualLearn)
    lbl7.parent:=FrmVisualLearn;
    lbl7.setbounds(800, 396, 62, 32);
  end;
  Img8:= TImage.create(FrmVisualLearn)
  with img8 do begin
   parent:= FrmVisualLearn;
   setbounds( 700, 432, 32, 32)
    Stretch := True
   lbl8:= Tlabel.create(FrmVisualLearn)
    lbl8.parent:=FrmVisualLearn;
    lbl8.setbounds(700, 496, 62, 32); 
  end;
  Img9:= TImage.create(FrmVisualLearn)
  with img9 do begin
   parent:= FrmVisualLearn;
   setbounds( 800, 432, 32, 32)
    Stretch := True
   lbl9:= Tlabel.create(FrmVisualLearn)
   lbl9.parent:=FrmVisualLearn;
   lbl9.setbounds(800, 496, 62, 32); 
  end;
  Img10:= TImage.create(FrmVisualLearn)
  with img10 do begin
   parent:= FrmVisualLearn;
   setbounds( 700, 532, 32, 32)
    Stretch := True
    lbl10:= Tlabel.create(FrmVisualLearn)
    lbl10.parent:=FrmVisualLearn;
    lbl10.setbounds(700, 596, 62, 32);
  end;
   LoadPicture2(self);
  //TestNNThread(0, 0,nil);
 end;
end; 

begin //@main

 setClassifierLabels;
 loadCifarForm();                           
 end.
End.


doc: https://maxbox4.wordpress.com/2022/07/07/cnn-pipeline-train/

https://www.cs.toronto.edu/~kriz/cifar.html

ref:    csMachineAnimalCifar10Labels: array[0..9] of string =
  (
    'airplane',
    'automobile',
    'ship',  // used to be bird
    'truck', // used to be cat
    'deer',  // used to be deer  ok
    'dog',   // used to be dog   ok
    'frog',  // used to be frog  ok
    'horse', // used to be horse ok
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


Creating CNeural Network...
Loading neural network from file: C:\maXbox\EKON_BASTA\EKON24\cifar-10-batches-bin\EKON25_SimpleImageClassifier-60.nn
Debug TNNet.Struct.LoadFromString ST: 
-1)TNNetInput:32;32;3;0;0;0;0;0
0)TNNetConvolutionLinear:64;5;2;1;1;0;0;0
1)TNNetMaxPool:4;4;0;0;0;0;0;0
2)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
3)TNNetConvolutionReLU:64;3;1;1;1;0;0;0
4)TNNetConvolutionReLU:64;3;1;1;1;0;0;0
5)TNNetConvolutionReLU:64;3;1;1;1;0;0;0
6)TNNetConvolutionReLU:64;3;1;1;1;0;0;0
7)TNNetDropout:2;1;0;0;0;0;0;0
8)TNNetMaxPool:2;2;0;0;0;0;0;0
9)TNNetFullConnectLinear:10;1;1;0;0;0;0;0
10)TNNetSoftMax:0;0;0;0;0;0;0;0

debug createLayer():TNNetInput-addLayerAfter-1...TNNetInput:32;32;3;0;0;0;0;0
debug createLayer():TNNetConvolutionLinear-addLayerAfter0...TNNetConvolutionLinear:64;5;2;1;1;0;0;0
debug createLayer():TNNetMaxPool-addLayerAfter1...TNNetMaxPool:4;4;0;0;0;0;0;0
debug createLayer():TNNetMovingStdNormalization-addLayerAfter2...TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter3...TNNetConvolutionReLU:64;3;1;1;1;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter4...TNNetConvolutionReLU:64;3;1;1;1;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter5...TNNetConvolutionReLU:64;3;1;1;1;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter6...TNNetConvolutionReLU:64;3;1;1;1;0;0;0
debug createLayer():TNNetDropout-addLayerAfter7...TNNetDropout:2;1;0;0;0;0;0;0
debug createLayer():TNNetMaxPool-addLayerAfter8...TNNetMaxPool:2;2;0;0;0;0;0;0
debug createLayer():TNNetFullConnectLinear-addLayerAfter9...TNNetFullConnectLinear:10;1;1;0;0;0;0;0
debug createLayer():TNNetSoftMax-addLayerAfter10...TNNetSoftMax:0;0;0;0;0;0;0;0
Debug TNNet Loaded Data Layers:12  SCount:12
Layer 0                            Max Output: 0 Min Output: 0 TNNetInput 32,32,3
 Times: 0s 0s
debug weights end.....
Layer 1 Neurons:64 Max Weight: 0.722684502601624 Min Weight: -0.760756492614746 Max Output: 0 Min Output: 0 TNNetConvolutionLinear 32,32,64
 Times: 0s 0s
 Parent:0
Layer 2                            Max Output: 0 Min Output: 0 TNNetMaxPool 8,8,64
 Times: 0s 0s
 Parent:1
Layer 3 Neurons:1 Max Weight: 0.977015614509583 Min Weight: 0.95193749666214 Max Output: 0 Min Output: 0 TNNetMovingStdNormalization 8,8,64
 Times: 0s 0s
 Parent:2
Layer 4 Neurons:64 Max Weight: 0.397520333528519 Min Weight: -0.417468905448914 Max Output: 0 Min Output: 0 TNNetConvolutionReLU 8,8,64
 Times: 0s 0s
 Parent:3
Layer 5 Neurons:64 Max Weight: 0.437700569629669 Min Weight: -0.391394764184952 Max Output: 0 Min Output: 0 TNNetConvolutionReLU 8,8,64
 Times: 0s 0s
 Parent:4
Layer 6 Neurons:64 Max Weight: 0.371823132038116 Min Weight: -0.30026838183403 Max Output: 0 Min Output: 0 TNNetConvolutionReLU 8,8,64
 Times: 0s 0s
 Parent:5
Layer 7 Neurons:64 Max Weight: 0.238006591796875 Min Weight: -0.249948143959045 Max Output: 0 Min Output: 0 TNNetConvolutionReLU 8,8,64
 Times: 0s 0s
 Parent:6
Layer 8                            Max Output: 0 Min Output: 0 TNNetDropout 8,8,64
 Times: 0s 0s
 Parent:7
Layer 9                            Max Output: 0 Min Output: 0 TNNetMaxPool 4,4,64
 Times: 0s 0s
 Parent:8
Layer 10 Neurons:10 Max Weight: 0.339100420475006 Min Weight: -0.267543852329254 Max Output: 0 Min Output: 0 TNNetFullConnectLinear 10,1,1
 Times: 0s 0s
 Parent:9
Layer 11                            Max Output: 0 Min Output: 0 TNNetSoftMax 10,1,1
 Times: 0s 0s
 Parent:10
Neural CNN network has: 
 Layers: 12
 Neurons: 331
 Weights: 162498
Computing...
Loading 10K images from file "data_batch_5.bin" ...
 GLOBAL MIN MAX -2  1.984375
 Done...
 Totalsize: 30720000
Ver: 4.7.6.10 (476). Workdir: C:\maXbox\EKON_BASTA\EKON24\cifar-10-batches-bin
Ver: 4.7.6.10 (476). Workdir: C:\maXbox\EKON_BASTA\EKON24\cifar-10-batches-bin
Layer 0                            Max Output: 1.359375 Min Output: -1.453125 TNNetInput 32,32,3
 Times: 0s 0s
debug weights end.....
Layer 1 Neurons:64 Max Weight: 0.722684502601624 Min Weight: -0.760756492614746 Max Output: 5.61273002624512 Min Output: -5.06748676300049 TNNetConvolutionLinear 32,32,64
 Times: 14.2969891894609s 0s
 Parent:0
Layer 2                            Max Output: 5.61273002624512 Min Output: -0.9996058344841 TNNetMaxPool 8,8,64
 Times: 0.290999980643392s 0s
 Parent:1
Layer 3 Neurons:1 Max Weight: 0.977015614509583 Min Weight: 0.95193749666214 Max Output: 4.86976766586304 Min Output: -2.07641935348511 TNNetMovingStdNormalization 8,8,64
 Times: 0.00799947883933783s 0s
 Parent:2
Layer 4 Neurons:64 Max Weight: 0.397520333528519 Min Weight: -0.417468905448914 Max Output: 10.0402660369873 Min Output: 0 TNNetConvolutionReLU 8,8,64
 Times: 8.27099843882024s 0s
 Parent:3
Layer 5 Neurons:64 Max Weight: 0.437700569629669 Min Weight: -0.391394764184952 Max Output: 15.7552042007446 Min Output: 0 TNNetConvolutionReLU 8,8,64
 Times: 7.44000063277781s 0s
 Parent:4
Layer 6 Neurons:64 Max Weight: 0.371823132038116 Min Weight: -0.30026838183403 Max Output: 11.411358833313 Min Output: 0 TNNetConvolutionReLU 8,8,64
 Times: 8.12000662554055s 0s
 Parent:5
Layer 7 Neurons:64 Max Weight: 0.238006591796875 Min Weight: -0.249948143959045 Max Output: 3.98265981674194 Min Output: 0 TNNetConvolutionReLU 8,8,64
 Times: 7.62000367976725s 0s
 Parent:6
Layer 8                            Max Output: 3.98265981674194 Min Output: 0 TNNetDropout 8,8,64
 Times: 0s 0s
 Parent:7
Layer 9                            Max Output: 3.98265981674194 Min Output: 0 TNNetMaxPool 4,4,64
 Times: 0s 0s
 Parent:8
Layer 10 Neurons:10 Max Weight: 0.339100420475006 Min Weight: -0.267543852329254 Max Output: 5.93420314788818 Min Output: -3.98547697067261 TNNetFullConnectLinear 10,1,1
 Times: 0.0410000793635845s 0s
 Parent:9
Layer 11                            Max Output: 0.95268440246582 Min Output: 4.68691323476378E-5 TNNetSoftMax 10,1,1
 Times: 0s 0s
 Parent:10
Test batch score:  Rate:0.8730, Loss:-0.3234, ErrorSum:963.6668 
closed & free action...
Test batch score:  Rate:0.9060, Loss:-0.3971, ErrorSum:967.4329 
closed & free action...
Ver: 4.7.6.20 (476). Workdir: C:\maXbox\EKON_BASTA\EKON24\cifar-10-batches-bin

Test batch score:  Rate:0.8630, Loss:-0.2911, ErrorSum:953.0551 
closed & free action...
Ver: 4.7.6.20 (476). Workdir: C:\maXbox\EKON_BASTA\EKON24\cifar-10-batches-bin
 Test batch score:  Rate:0.8890, Loss:-0.3462, ErrorSum:967.9903 
closed & free action...
Ver: 4.7.6.20 (476). Workdir: C:\maXbox\EKON_BASTA\EKON24\cifar-10-batches-bin