program IdentityShortcutConnection_mX4;
(*
 Coded by Joao Paulo Schwarz Schuler.
 https://github.com/joaopauloschuler/neural-api
*)
//{$mode objfpc}{$H+}

(*
uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, CustApp, neuralnetwork, neuralvolume, Math, neuraldatasets,
  neuralfit, neuralopencl;    *)

type
  TTestCNNAlgo = {class(}TCustomApplication;
  //protected
    procedure DoRun; forward;//override;
  //end;

  procedure TTestCNNAlgoDoRun;
  var
    NN: TNNet;
    NeuralFit: TNeuralImageFit;
    ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes: TNNetVolumeList;
    GlueLayer: TNNetLayer;
    //EasyOpenCL: TEasyOpenCL;
  begin
    if not CheckCIFARFile() then
    begin
      application.Terminate;
      exit;
    end;

    WriteLn('Creating Neural Network...');
    NN := TNNet.Create();
    GlueLayer := NN.AddLayer2([
      TNNetInput.Create4(32, 32, 3),
      TNNetConvolutionLinear.Create({NumFeatures=}64, 
      {featureSize=}5, {padding=}2, {stride=}1, 
      {SuppressBias=}0).InitBasicPatterns()
    ]);

  (*  NN.AddLayer(TNNetConvolutionReLU.Create({NumFeatures=}64 
          ,{featureSize=}3, {padding=}1, {stride=}1, {SuppressBias=}0));
    NN.AddLayer(TNNetConvolutionLinear.Create({NumFeatures=}64 ,
          {featureSize=}3, {padding=}1, {stride=}1, {SuppressBias=}0));
    NN.AddLayer(TNNetSum.Create21([NN.GetLastLayer(), GlueLayer]));
    //RegisterMethod('Constructor Create( LowValue, HighValue : integer);'); bug
    //GlueLayer := NN.AddLayer(TNNetReLUBase.Create(0,1));
    GlueLayer := NN.AddLayer(TNNetSoftMax.Create());

    NN.AddLayer(TNNetConvolutionReLU.Create({NumFeatures=}64 ,{featureSize=}3, {padding=}1, {stride=}1, {SuppressBias=}0));
    NN.AddLayer(TNNetConvolutionLinear.Create({NumFeatures=}64 ,{featureSize=}3, {padding=}1, {stride=}1, {SuppressBias=}0));
    
    NN.AddLayer(TNNetSum.Create21([NN.GetLastLayer(), GlueLayer]));
    
    *)
    //NN.AddLayer(TNNetReLU.Create(0,1));
    //NN.AddLayer(TNNetSigmoid.Create());
    //NN.AddLayer(TNNetSeLU.Create());
       NN.AddLayer(TNNetReLUL.Create8(0,1));

    NN.AddLayer2([
      TNNetDropout.Create12(0.5,1),
      TNNetMaxPool.Create44(4,0,0),
      TNNetFullConnectLinear.Create28(10,0),
      TNNetSoftMax.Create()
    ]);
    //*)
    NN.DebugWeights();
    NN.DebugStructure();
    WriteLn('Layers: '+itoa( NN.CountLayers()));
    WriteLn('Neurons: '+itoa( NN.CountNeurons()));
    WriteLn('Weights: '+itoa( NN.CountWeights()));

    CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes, 
                           ImgTestVolumes, csEncodeRGB);

    NeuralFit := TNeuralImageFit.Create;
    NeuralFit.FileNameBase := 'SimpleImageClassifierIDShortcut2';
    NeuralFit.InitialLearningRate := 0.001;
    NeuralFit.LearningRateDecay := 0.01;
    NeuralFit.StaircaseEpochs := 10;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0.00001;

 (*   EasyOpenCL := TEasyOpenCL.Create();
    if EasyOpenCL.GetPlatformCount() > 0 then
    begin
      WriteLn('Setting platform to: ', EasyOpenCL.PlatformNames[0]);
      EasyOpenCL.SetCurrentPlatform(EasyOpenCL.PlatformIds[0]);
      if EasyOpenCL.GetDeviceCount() > 0 then
      begin
        EasyOpenCL.SetCurrentDevice(EasyOpenCL.Devices[0]);
        WriteLn('Setting device to: ', EasyOpenCL.DeviceNames[0]);
        NeuralFit.EnableOpenCL(EasyOpenCL.PlatformIds[0], EasyOpenCL.Devices[0]);
      end
      else
      begin
        WriteLn('No OpenCL capable device has been found for platform ',EasyOpenCL.PlatformNames[0]);
        WriteLn('Falling back to CPU.');
      end;
    end
    else
    begin
      WriteLn('No OpenCL platform has been found. Falling back to CPU.');
    end;    *)

    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, 
           ImgTestVolumes, {NumClasses=}10, {batchsize=}128, {epochs=}5); //->50
    NeuralFit.Free;

    NN.Free;
    ImgTestVolumes.Free;
    ImgValidationVolumes.Free;
    ImgTrainingVolumes.Free;
    //application.Terminate;
  end; //$
  
  procedure DoRun; 
  begin
    TTestCNNAlgoDoRun;
  end;  

var
  nnApplication: TTestCNNAlgo;
begin
  nnApplication := TTestCNNAlgo.Create(nil);
  nnApplication.Title:='CIFAR-10 Identity Shortcut Connection';
  //nnApplication.Run;
  DoRun;
  nnApplication.Free;
End.


TNNetMaxPool:4;4;0;0;0;0;0;0#4)TNNetFullConnectLinear:10;1;1;0;0;0;0;0#5)TNNetSoftMax:0;0;0;0;0;0;0;0
Epochs: 1 Examples seen:40000 Validation Accuracy: 0.5448 Validation Error: 1.1514 Validation Loss: 1.3314 Total time: 19.63min
Image mX4 FThreadNN[0].DebugWeights(); skipped...
Epoch time: 11.0000 minutes. 1 epochs: 0.1900 hours.
Epochs: 1. Working time: 0.33 hours.
CAI maXbox Neural Fit Finished.
 mX4 executed: 05/01/2022 23:06:29  Runtime: 0:19:44.163  Memload: 45% use
