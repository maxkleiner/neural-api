program Cifar10Resize_mX4_2_LoadModel;
(*
 Coded by Joao Paulo Schwarz Schuler.
 https://github.com/joaopauloschuler/neural-api
 https://github.com/maxkleiner/neural-api/blob/master/examples/Cifar10Resize/Cifar10Resize.lpr
 https://github.com/maxkleiner/neural-api/blob/master/neural/neuralthread.pas
 https://sourceforge.net/p/cai/svncode/1313/tree/trunk/lazarus/examples/SuperResolution/SuperResolution.lpr
 http://www.softwareschule.ch/examples/uPSI_NeuralNetworkCAI.txt
 http://www.softwareschule.ch/examples/uPSI_neuralvolume.txt
 https://sourceforge.net/p/cai/svncode/HEAD/tree/trunk/lazarus/experiments/
 https://sourceforge.net/p/cai/svncode/HEAD/tree/trunk/lazarus/experiments/visualCifar10test/uvisualcifar10test.pas#l151
*)
//{$mode objfpc}{$H+}

(*                                                         
uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, CustApp, neuralnetwork, neuralvolume,
  Math, neuraldatasets, neuralfit, neuralthread, usuperresolutionexample;

  *)
  
Const PReModel_NN = 'SimpleSeparableImageClassifier124_50_2.nn'; 
  
type
  TCifar10Resize = {class(}TCustomApplication;
  //protected
  var
    FFolderName: string;                                
    FCIFARType: string; //10 or 100
    FCIFARClassCount: integer; //10 or 100
    FBaseCount: integer;
    FThreadList: TNeuralThreadList;
    FCurrentVolumeList: TNNetVolumeList;
    FCriticalSection: TRTLCriticalSection;
    procedure TCifar10ResizeDoRun; forward; //override;
    procedure ProcessVolumeList(VolumeList: TNNetVolumeList; FolderName: string; BaseCount: integer); forward;
    procedure ResizeImages_NTL(index, threadnum: integer); forward;
    procedure CreateCifarDirectories(pCifarTypeStr: string; pCifarClassCount: integer); forward;
  //end;

  procedure CreateDirectories(FolderName: string; ClassCount:integer);
  var
    ClassCnt: integer;
  begin
    for ClassCnt := 0 to ClassCount - 1 do begin
      writeln(botostr(ForceDirectories(FolderName+'\class'+IntToStr(ClassCnt))));
     writeln('forcedir: '+FolderName+'\class'+IntToStr(ClassCnt)) 
    end;
    //Exception: Unable to create directory.
  end;

  procedure ProcessVolumeList(VolumeList:TNNetVolumeList; FolderName:string; BaseCount: integer);
  begin
    FFolderName := FolderName;
    FBaseCount := BaseCount;
    FCurrentVolumeList := VolumeList;
    //procedure StartProc(pProc: TNeuralProc; pBlock: boolean = true);
    FThreadList.StartProc(@ResizeImages_NTL, true);
    //ResizeImages_NTL(1,1)
  end;
  
  
 Const
  csExampleBaseFileName = 'super-resolution-cifar-10';
  csExampleFileName{:string} = 'super-resolution-cifar-10.nn';
  csExampleNeuronCount = 64;
  csExampleLayerCount = 7;
  
  
procedure LoadResizingWeights(NN: TNNet; FileName: string; FailIfNotFound: boolean {= false});
begin
  if FileExists(FileName) then begin
    NN.LoadDataFromFile(FileName);
    WriteLn('Example file found:'+ FileName);
  end else
  if FileExists('../../../examples/SuperResolution/'+FileName) then begin
    NN.LoadDataFromFile('../../../examples/SuperResolution/'+FileName);
    WriteLn('Example file found at ../../../examples/SuperResolution : '+ FileName);
  end else
  begin if FailIfNotFound then begin
      WriteLn('ERROR: '+FileName+' can''t be found. Please run SuperResolutionTrain.');
      //ReadLn;
    end;
  end;
end;
 
 //function THistoricalNets.AddSuperResolution(pSizeX, pSizeY, BottleNeck, pNeurons,
  //pLayerCnt: integer; IsSeparable:boolean): TNNetLayer; 
  
function CreateResizingNN(SizeX, SizeY: integer; FileName: string): THistoricalNets;
var
  NN: THistoricalNets;
begin
  NN := THistoricalNets.Create();
  NN.AddSuperResolution({pSizeX=}SizeX, {pSizeY=}SizeY, 0, {pNeurons=}csExampleNeuronCount, {pLayerCnt=}csExampleLayerCount, true);
  LoadResizingWeights(NN, FileName, true);
  Result := NN;
end;

  procedure ResizeImages_NTL(index, threadnum: integer);
  var
    ImgCnt: integer;
    CurrentImg: TNNetVolume;
    ClassId: integer;
    StartPos, FinishPos: integer;
    NN32to64: THistoricalNets;
    NN64to128: THistoricalNets;
    Aux64Vol: TNNetVolume;
    Aux128Vol: TNNetVolume;
  begin
    EnterCriticalSection(FCriticalSection);
    //WriteLn('Creating neural networks at thread ',index,'.');
    WriteLn('Creating neural networks at thread '+itoa(index)+'.');
    //https://stackoverflow.com/questions/41102410/image-resizing-method-during-preprocessing-for-neural-network

    NN32to64 := CreateResizingNN(32, 32, csExampleFileName);
    NN64to128 := CreateResizingNN(64, 64, csExampleFileName);
    Aux64Vol := TNNetVolume.Create();
    Aux128Vol := TNNetVolume.Create();
    FThreadList.CalculateWorkingRange(index, threadnum, FCurrentVolumeList.Count,
      StartPos, FinishPos);
   WriteLn('Thread '+itoa(index)+' has working range from '+itoa(StartPos)+' to '+
                   itoa(FinishPos)+'. This thread is now starting.');
    LeaveCriticalSection(FCriticalSection);
    for ImgCnt := StartPos to FinishPos do begin
      if ImgCnt mod 100 = 0 then begin
       WriteLn('Thread '+itoa(index)+' is processing image '+itoa( FBaseCount) + 
                           itoa(ImgCnt)+' at '+ FFolderName+'.');  end;
      CurrentImg := FCurrentVolumeList[ImgCnt];
      ClassId := CurrentImg.Tag;
      NN32to64.Compute65(CurrentImg,0);
      NN32to64.GetOutput(Aux64Vol);
      NN64to128.Compute65(Aux64Vol,0);
      NN64to128.GetOutput(Aux128Vol);
      CurrentImg.Add(2);
      CurrentImg.Mul67(64);
      Aux64Vol.Add(2);
      Aux64Vol.Mul67(64);
      Aux128Vol.Add(2);
      Aux128Vol.Mul67(64);
      SaveImageFromVolumeIntoFile(CurrentImg,
        'resized/cifar'+FCIFARType+'-32/'+FFolderName+'/class'+IntToStr(ClassId)+'/'+
        'img'+IntToStr(FBaseCount+ImgCnt)+'.png');
      SaveImageFromVolumeIntoFile(Aux64Vol,
        'resized/cifar'+FCIFARType+'-64/'+FFolderName+'/class'+IntToStr(ClassId)+'/'+
        'img'+IntToStr(FBaseCount+ImgCnt)+'.png');
      SaveImageFromVolumeIntoFile(Aux128Vol,
        'resized/cifar'+FCIFARType+'-128/'+FFolderName+'/class'+IntToStr(ClassId)+'/'+
        'img'+IntToStr(FBaseCount+ImgCnt)+'.png');
    end;
    WriteLn('Thread '+itoa(index)+' has finished.');
    Aux128Vol.Free;
    Aux64Vol.Free;
    NN64to128.Free;
    NN32to64.Free;
  end;

  procedure CreateCifarDirectories(pCifarTypeStr: string;
    pCifarClassCount: integer);
  begin
    CreateDirectories(exepath+'resized\cifar'+pCifarTypeStr+'-32\train',pCifarClassCount);
    CreateDirectories(exepath+'resized\cifar'+pCifarTypeStr+'-64\train',pCifarClassCount);
    CreateDirectories(exepath+'resized\cifar'+pCifarTypeStr+'-128\train',pCifarClassCount);
    CreateDirectories(exepath+'resized\cifar'+pCifarTypeStr+'-32\train',pCifarClassCount);
    CreateDirectories(exepath+'resized\cifar'+pCifarTypeStr+'-64\train',pCifarClassCount);
    CreateDirectories(exepath+'resized\cifar'+pCifarTypeStr+'-128\train',pCifarClassCount);
    CreateDirectories(exepath+'resized\cifar'+pCifarTypeStr+'-32\test',pCifarClassCount);
    CreateDirectories(exepath+'resized\cifar'+pCifarTypeStr+'-64\test',pCifarClassCount);
    CreateDirectories(exepath+'resized\cifar'+pCifarTypeStr+'-128\test',pCifarClassCount);
    CreateDirectories(exepath+'resized\cifar'+pCifarTypeStr+'-32\test',pCifarClassCount);
    CreateDirectories(exepath+'resized\cifar'+pCifarTypeStr+'-64\test',pCifarClassCount);
    CreateDirectories(exepath+'resized\cifar'+pCifarTypeStr+'-128\test',pCifarClassCount);  //}
  end;

  procedure TCifar10ResizeDoRun;
  var
    ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes: TNNetVolumeList;
    IsCifar100: boolean;    ProportionToLoad: Single;
     FTrainingFileNames, FValidationFileNames, FTestFileNames: TFileNameList;
  begin
    if not CheckCIFARFile() then begin
      //Terminate;
      //exit;
      writeln('___terminate & exit___');
    end;
    IsCifar100 := false; // change this if you need creating CIFAR-100 resized.

    WriteLn('Creating Threads');
    InitCriticalSection(FCriticalSection);
    FThreadList := TNeuralThreadList.Create( NeuralDefaultThreadCount() );

    if IsCifar100 then begin
      FCIFARType := '100';
      FCIFARClassCount := 100;
      CreateCifar100Volumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, 
          csEncodeRGB, true);
    end else begin
      FCIFARType := '10';
      FCIFARClassCount := 10;
      
      // creates CIFAR10 volumes required for training, testing and validation
//procedure CreateCifar10Volumes(out ImgTrainingVolumes, ImgValidationVolumes,
  //ImgTestVolumes: TNNetVolumeList; color_encoding: byte = csEncodeRGB;
  //ValidationSampleSize: integer = 2000);
  //    no ValidationSampleSize in this interface: 
      CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, 
                  csEncodeRGB    );
    end;
    CreateCifarDirectories(FCIFARType, FCIFARClassCount);
    
     ProportionToLoad := 1;
  (*   CreateVolumesFromImagesFromFolder
    (
      ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes,
      {FolderName=}'plant', {pImageSubFolder=}'',
      {color_encoding=}0{RGB},
      {TrainingProp=}0.9*ProportionToLoad,
      {ValidationProp=}0.05*ProportionToLoad,
      {TestProp=}0.05*ProportionToLoad,
      {NewSizeX=}128, {NewSizeY=}128
    );
    
    CreateFileNameListsFromImagesFromFolder(
      FTrainingFileNames, FValidationFileNames, FTestFileNames,
      {FolderName=}'plant', {pImageSubFolder=}'',
      {TrainingProp=}0.9*ProportionToLoad,
      {ValidationProp=}0.05*ProportionToLoad,
      {TestProp=}0.05*ProportionToLoad
    ); *)

    ProcessVolumeList(ImgTrainingVolumes, 'train', 0);
    ProcessVolumeList(ImgValidationVolumes, 'train', 40000);
    ProcessVolumeList(ImgTestVolumes, 'test', 0);
    ImgTestVolumes.Free;
    ImgValidationVolumes.Free;
    ImgTrainingVolumes.Free;

    DoneCriticalSection(FCriticalSection);
    FThreadList.Free();
    //Terminate;
    writeln('__terminate__');
  end;
  
procedure LoadPreTrainModel(Sender: TObject);
var filename: string;
    FNN: TNNet;
begin
(*  if (OpenDialogNN.Execute()) then begin
    if FileExists(OpenDialogNN.FileName) then begin
      ButLearn.Enabled := false;
      FileName := OpenDialogNN.FileName;
      if not(FileExists(FileName)) then begin
        if FileName = 'SimpleImageClassifier.nn' then begin
          WriteLn('Please run the Simple Image Classifier example before running this example.');
          WriteLn('https://github.com/joaopauloschuler/neural-api/tree/master/examples/SimpleImageClassifier');
        end
        else begin
          WriteLn('File not found:'+FileName);
        end;
        exit;
      end;
    end;
  end;  *) 
  FNN:= TNNet.Create();
  try
     Filename:= Exepath+PReModel_NN;
      FNN.LoadFromFile(FileName);
      FNN.DebugStructure();
      FNN.DebugWeights();
      FNN.SetBatchUpdate( true );
      FNN.SetLearningRate(0.01,0.0);
      TNNetInput(FNN.Layers[0]).EnableErrorCollection();
      WriteLn('Neural CNN network has: ');
      WriteLn(' Layers: '+ itoa(FNN.CountLayers()  ));
      WriteLn(' Neurons: '+ itoa(FNN.CountNeurons() ));
      WriteLn(' Weights: '+ itoa(FNN.CountWeights() ));
  finally
    FNN.Free;
  end;   
      //ButLearn.Enabled := true;
      //LoadNNLayersIntoCombo(FNN, ComboLayer);
    //end;
  //end;
end;
 
procedure LoadCompute_PreTrainModel(apmodel: string; visuallearnForm: TForm);
var filename: string;
    FNN: TNNet;
    ImgVolumes: TNNetVolumeList;
    Volume: TNNetVolume;
    pOutput, vOutput, vDisplay: TNNetVolume;
    NumClasses, i: integer;
    firstNeuronalLayer, NeuronCount: integer;
    aImage: array of TImage; ab: TBitmap;
    Rate, Loss, ErrorSum: TNeuralFloat;
    
begin
(*  if (OpenDialogNN.Execute()) then begin
    if FileExists(OpenDialogNN.FileName) then begin
      ButLearn.Enabled := false;
      FileName := OpenDialogNN.FileName;
      if not(FileExists(FileName)) then begin
        if FileName = 'SimpleImageClassifier.nn' then begin
          WriteLn('Please run the Simple Image Classifier example before running this example.');
          WriteLn('https://github.com/joaopauloschuler/neural-api/tree/master/examples/SimpleImageClassifier');
        end
        else begin
          WriteLn('File not found:'+FileName);
        end;
        exit;
      end;
    end;
  end;  *) 
  FNN:= TNNet.Create();
  try
    Filename:= apModel; //Exepath+PReModel_NN;
    FNN.LoadFromFile(FileName);
    FNN.EnableDropouts(false);
    FNN.DebugStructure();
    FNN.DebugWeights();
    FNN.SetBatchUpdate( true );
    FNN.SetLearningRate(0.01,0.0);
    TNNetInput(FNN.Layers[0]).EnableErrorCollection();
    WriteLn('Neural CNN network has: ');
    WriteLn(' Layers: '+ itoa(FNN.CountLayers()  ));
    WriteLn(' Neurons: '+ itoa(FNN.CountNeurons() ));
    WriteLn(' Weights: '+ itoa(FNN.CountWeights() ));
      
    writeln('Creating CNeural Network...');
    ImgVolumes := TNNetVolumeList.Create(true);
    NumClasses := 10;
     
       // creates required volumes to store images
  for i:= 0 to 9999 do begin
    Volume := TNNetVolume.Create();
    ImgVolumes.Add(Volume);
  end;
  
  //testmap
  ab:= TBitmap.create;
  ab.loadfromresourcename(0,'MOON_FULL'); //testpic
  
  firstNeuronalLayer:= FNN.GetFirstNeuronalLayerIdx(0);
  writeln('firstNeuronalLayer: '+itoa(firstNeuronalLayer));
  
  pOutput := TNNetVolume.Create0(NumClasses,1,1,1);
  vOutput := TNNetVolume.Create0(NumClasses,1,1,1);
  vDisplay:= TNNetVolume.Create0(NumClasses,1,1,1);
  writeln('1.Neuron Count: '+itoa(FNN.Layers[firstNeuronalLayer].Neurons.Count));
  SetLength(aImage, FNN.Layers[firstNeuronalLayer].Neurons.Count);

  //writeln('1.Neuron Count: '+itoa(FNN.Layers[firstNeuronalLayer].Neurons.Count));
  for NeuronCount:= 0 to FNN.Layers[firstNeuronalLayer].Neurons.Count-1 do begin
    aImage[NeuronCount]:= TImage.Create(visuallearnForm);
    aImage[NeuronCount].Parent:= visuallearnForm;
    aImage[NeuronCount].Width:= 
                          FNN.Layers[firstNeuronalLayer].Neurons[NeuronCount].Weights.SizeX+10;
    aImage[NeuronCount].Height:= 
                          FNN.Layers[firstNeuronalLayer].Neurons[NeuronCount].Weights.SizeY+10;
    //writeln('debug: '+itoa(aImage[NeuronCount].Width));      
    //writeln('debug: '+itoa(aImage[NeuronCount].Height));                      
    aImage[NeuronCount].Top:= (NeuronCount div 12) * 36 + 120;
    aImage[NeuronCount].Left:= (NeuronCount mod 12) * 36 + 32;
    // writeln('debug: '+itoa(aImage[NeuronCount].top));  
    //  writeln('debug: '+itoa(aImage[NeuronCount].left));  
    aImage[NeuronCount].Stretch:=true;
    //aImage[NeuronCount].color:=clYellow;
    aImage[NeuronCount].picture.bitmap:= ab;
     aImage[NeuronCount].visible:= true;
  end;
   visuallearnForm.Show;
   
   WriteLn('Neural CNN network has second test: ');
    WriteLn(' Layers: '+ itoa(FNN.CountLayers()  ));
    WriteLn(' Neurons: '+ itoa(FNN.CountNeurons() ));
    WriteLn(' Weights: '+ itoa(FNN.CountWeights() ));
    
  if CheckCIFARFile() then begin
      //Terminate;
      //exit;
      writeln('___cifar datasets found___');
      
      // loads a CIFAR10 into TNNetVolumeList
//procedure loadCifar10Dataset(ImgVolumes: TNNetVolumeList; idx:integer; base_pos:integer = 0; color_encoding: byte = csEncodeRGB); overload;
//procedure loadCifar10Dataset(ImgVolumes: TNNetVolumeList; fileName:string; base_pos:integer = 0; color_encoding: byte = csEncodeRGB); overload;
    
    loadCifar10Dataset2(ImgVolumes, 2, 0, csEncodeRGB);
    WriteLn(' Totalsize: '+ itoa(ImgVolumes.gettotalsize));
    
    //loadCifar10Dataset(ImgVolumes, exepath+'data_batch_2.bin', 0, csEncodeRGB);
    //WriteLn(' Totalsize: '+ itoa(ImgVolumes.gettotalsize));
    //sleep(5000)
   // This function tests a neural network on the passed ImgVolumes
 {procedure TestBatch
 (
  NN: TNNet; ImgVolumes: TNNetVolumeList; SampleSize: integer;
  out Rate, Loss, ErrorSum: TNeuralFloat
 ); }
    rate := 0;
    loss := 0;
    ErrorSum := 0;
    TestBatch(FNN, ImgVolumes, 1000, Rate, Loss, ErrorSum);
    writeln('Test batch score: '+Format(' Rate:%.4f, Loss:%.4f, ErrorSum:%.4f ',
                                          [Rate, Loss, ErrorSum]));
  end;

  finally
    for NeuronCount := Low(aImage) to High(aImage) do begin
      aImage[NeuronCount].Free;       
    end;

    ab.Free;
    FNN.Free;
    ImgVolumes.Free;
    pOutput.Free;
    vOutput.Free;
    vDisplay.Free;
  end;   
      //ButLearn.Enabled := true;
      //LoadNNLayersIntoCombo(FNN, ComboLayer);
    //end;
  //end;
end;

var
  Application2: TCifar10Resize;  afrm: TForm;
  fImage: array[0..2] of TImage;
  imgdirect:TImage; ab: TBitmap;
begin //@main
  Application2 := TCifar10Resize.Create(nil);
  Application2.Title:='CIFAR-10 LoadModel Example';
  afrm:= TForm.create(self);
  afrm.setbounds(0,0,900,440)
  afrm.Caption:= 'Cifar10VolumesToMachineShow maXbox4';
  afrm.icon.loadfromresourcename(hinstance,'XDIRECTX');
  //afrm.show; 
  //Application2.Run;
  //ShowmessageBig('uncomment the following line - TCifar10ResizeDoRun..., makes  files');
  //TCifar10ResizeDoRun;
  //LoadPreTrainModel(self);
  ab:= TBitmap.create;
  ab.loadfromresourcename(0,'MOON_FULL');  
  LoadCompute_PreTrainModel(exepath+PReModel_NN, aFrm);
    fImage[0]:= TImage.Create(afrm);
    fImage[0].Parent:= afrm;
    fImage[0].setbounds(10,10,200,200);
    fImage[0].picture.bitmap:= ab;
    
  {ab:= TBitmap.create;
  ab.loadfromresourcename(0,'MOON_FULL');  
  imgdirect:= TImage.create(self);
  with imgdirect do begin
   parent:= afrm;
    Left := 5; Height := 142;
    Top := 120; Width := 142;
    color:= clblue;
    Stretch := True
    picture.bitmap:= ab;
  end; } 
  //afrm.Show;
  ab.Free;
  Application2.Free;
End.

Ref: with test batch (TCifar10ResizeDoRun) creates 69 folders

 Directory of C:\maXbox\EKON_BASTA\EKON24\cifar-10-batches-bin\resized

28/06/2022  15:50    <DIR>          .
28/06/2022  15:50    <DIR>          ..
28/06/2022  15:50    <DIR>          cifar10-128
28/06/2022  15:47    <DIR>          cifar10-32
28/06/2022  15:50    <DIR>          cifar10-64

 Directory of C:\maXbox\EKON_BASTA\EKON24\cifar-10-batches-bin\resized\cifar10-1
28

28/06/2022  15:50    <DIR>          .
28/06/2022  15:50    <DIR>          ..
28/06/2022  15:50    <DIR>          test
28/06/2022  15:50    <DIR>          train
               0 File(s)              0 bytes

 Directory of C:\maXbox\EKON_BASTA\EKON24\cifar-10-batches-bin\resized\cifar10-6
4\train

28/06/2022  15:50    <DIR>          .
28/06/2022  15:50    <DIR>          ..
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
               0 File(s)              0 bytes


Doc: https://sourceforge.net/p/cai/svncode/1313/tree/trunk/lazarus/examples/SuperResolution/usuperresolutionexample.pas#l19
