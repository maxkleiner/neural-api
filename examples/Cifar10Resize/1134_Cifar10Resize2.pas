program Cifar10Resize_mX4_2;
(*
 Coded by Joao Paulo Schwarz Schuler.
 https://github.com/joaopauloschuler/neural-api
 https://github.com/maxkleiner/neural-api/blob/master/examples/Cifar10Resize/Cifar10Resize.lpr
*)
//{$mode objfpc}{$H+}

(*
uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, CustApp, neuralnetwork, neuralvolume,
  Math, neuraldatasets, neuralfit, neuralthread, usuperresolutionexample;

  *)
  
  
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
    procedure ProcessVolumeList(VolumeList: TNNetVolumeList; FolderName: string; 
                                           BaseCount: integer); forward;
    procedure ResizeImages_NTL(index, threadnum: integer); forward;
    procedure CreateCifarDirectories(pCifarTypeStr: string; pCifarClassCount: integer); forward;
  //end;

  procedure CreateDirectories(FolderName: string; ClassCount:integer);
  var
    ClassCnt: integer;
  begin
    for ClassCnt:= 0 to ClassCount - 1 do begin
      writeln(botostr(ForceDirectories(FolderName+'\class'+IntToStr(ClassCnt))));
    end;
  end;

  var idx: byte;
  procedure ProcessVolumeList(VolumeList:TNNetVolumeList; FolderName:string; 
                                     BaseCount: integer);
  begin
    FFolderName := FolderName;
    FBaseCount := BaseCount;
    FCurrentVolumeList := VolumeList;
    //procedure StartProc(pProc: TNeuralProc; pBlock: boolean = true);
    FThreadList.StartProc(@ResizeImages_NTL, true);
    ResizeImages_NTL(0,1);   
    inc(idx);            
    writeln('Resizer started ..');
  end;
  
  
 Const
  csExampleBaseFileName = 'super-resolution-cifar-10';
  csExampleFileName{:string} = 'super-resolution-cifar-10.nn';
  csExampleNeuronCount = 64;
  csExampleLayerCount = 7;
  
  
procedure LoadResizingWeights(NN: TNNet; FileName:string; FailIfNotFound: boolean{= false});
begin
  if FileExists(FileName) then begin
    NN.LoadDataFromFile(FileName);
    WriteLn('Example file found:'+ FileName);
  end else
  if FileExists('./examples/SuperResolution/'+FileName) then begin      
    NN.LoadDataFromFile('./examples/SuperResolution/'+FileName);
    WriteLn('Example file found at ./examples/SuperResolution : '+ FileName);
  end else begin
    if FailIfNotFound then begin
      WriteLn('ERROR: '+FileName+' can''t be found. Please run SuperResolutionTrain.');
      //ReadLn;
    end;
  end;
end;
 
 //function THistoricalNets.AddSuperResolution(pSizeX, pSizeY, BottleNeck, pNeurons,
  //pLayerCnt: integer; IsSeparable:boolean): TNNetLayer; 
  
function CreateResizingNN(SizeX, SizeY: integer; FileName: string): THistoricalNets;
var NN: THistoricalNets;
begin
  NN:= THistoricalNets.Create();
  NN.AddSuperResolution({pSizeX=}SizeX, {pSizeY=}SizeY,0, 
        {pNeurons=}csExampleNeuronCount, {pLayerCnt=}csExampleLayerCount, true);
  LoadResizingWeights(NN, FileName, true);
  Result:= NN;
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

    NN32to64:= CreateResizingNN(32, 32, csExampleFileName);
    NN64to128:= CreateResizingNN(64, 64, csExampleFileName);
    Aux64Vol:= TNNetVolume.Create();
    Aux128Vol:= TNNetVolume.Create();
    FThreadList.CalculateWorkingRange(index, threadnum, FCurrentVolumeList.Count,
                                            StartPos, FinishPos);
   WriteLn('Thread '+itoa(index)+' has working range from '+itoa(StartPos)+' to '+
                   itoa(FinishPos)+'. This thread is now starting.');
    LeaveCriticalSection(FCriticalSection);
    for ImgCnt:= StartPos to FinishPos do begin
      if ImgCnt mod 100 = 0 then begin
       WriteLn('Thread '+itoa(index)+' is processing image '+itoa(FBaseCount) + 
                           itoa(ImgCnt)+' at '+ FFolderName+'.');      end;
      CurrentImg:= FCurrentVolumeList[ImgCnt];
      ClassId:= CurrentImg.Tag;
      //procedure Compute(pInput: TNNetVolume; FromLayerIdx:integer = 0); overload;
      NN32to64.Compute65(CurrentImg,0);
      NN32to64.GetOutput(Aux64Vol);
      NN64to128.Compute65(Aux64Vol,0);
      NN64to128.GetOutput(Aux128Vol);
      //RegisterMethod('Procedure Add13( Value : TNeuralFloat);');
      CurrentImg.Add13(2);
      CurrentImg.Mul67(64);
      Aux64Vol.Add13(2);
      Aux64Vol.Mul67(64);
      Aux128Vol.Add13(2);
      Aux128Vol.Mul67(64);
      SaveImageFromVolumeIntoFile(CurrentImg,
        'resized\cifar'+FCIFARType+'-32\'+FFolderName+'\class'+IntToStr(ClassId)+'\'+
        'img'+IntToStr(FBaseCount+ImgCnt)+'.png');
      SaveImageFromVolumeIntoFile(Aux64Vol,
        'resized\cifar'+FCIFARType+'-64\'+FFolderName+'\class'+IntToStr(ClassId)+'\'+
        'img'+IntToStr(FBaseCount+ImgCnt)+'.png');
      SaveImageFromVolumeIntoFile(Aux128Vol,
        'resized\cifar'+FCIFARType+'-128\'+FFolderName+'\class'+IntToStr(ClassId)+'\'+
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
    CreateDirectories(exepath+'resized\cifar'+pCifarTypeStr+'-128\test',pCifarClassCount);
  end;

  procedure TCifar10ResizeDoRun;
  var
    ImgTrainingVolumes,ImgValidationVolumes,ImgTestVolumes: TNNetVolumeList;
    IsCifar100: boolean;
  begin
    if not CheckCIFARFile() then begin
      //Terminate;
      //exit;
      writeln('___terminate & exit___missing CIFAR');
    end;
    IsCifar100:= false; // change this if you need creating CIFAR-100 resized.
    idx:= 0;

    WriteLn('Creating Threads');
    InitCriticalSection(FCriticalSection);
    FThreadList:= TNeuralThreadList.Create( NeuralDefaultThreadCount() );

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
                                           csEncodeRGB);
    end;
    CreateCifarDirectories(FCIFARType, FCIFARClassCount);

    ProcessVolumeList(ImgTrainingVolumes, 'train', 0);
    ProcessVolumeList(ImgValidationVolumes, 'train', 40000);
    ProcessVolumeList(ImgTestVolumes, 'test', 0);
    ImgTestVolumes.Free;
    ImgValidationVolumes.Free;
    ImgTrainingVolumes.Free;

    DoneCriticalSection(FCriticalSection);
    FThreadList.Free();
    //Terminate;
  end;

var
  Application2: TCifar10Resize;
begin //@main
  Application2 := TCifar10Resize.Create(nil);
  Application2.Title:='CIFAR-10 Resizing Example';
  //Application2.Run;
  showMessageBig('uncomment the line below - TCifar10ResizeDoRun...');
  TCifar10ResizeDoRun;
  Application2.Free;
End.

Doc: https://sourceforge.net/p/cai/svncode/1313/tree/trunk/lazarus/examples/SuperResolution/usuperresolutionexample.pas#l19
