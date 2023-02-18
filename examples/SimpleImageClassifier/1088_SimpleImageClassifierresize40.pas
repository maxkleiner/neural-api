program SimpleImageClassifierSharedWeights_mX4_size48;
(*
 Coded by Joao Paulo Schwarz Schuler.
 https://github.com/joaopauloschuler/neural-api
*)
//{$mode objfpc}{$H+}
(*
uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}                                                          
  Classes, SysUtils, CustApp, neuralnetwork, neuralvolume, Math, neuraldatasets, neuralfit; *)

type
  TTestCNNAlgo = {class(}TCustomApplication;
  //protected
    procedure TTestCNNAlgoDoRun; forward;//override;
  //end;

  procedure TTestCNNAlgoDoRun;
  var
    NN: TNNet;
    NeuralFit: TNeuralImageFit;
    ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes: TNNetVolumeList;
    FirstBranch, SecondBranch: TNNetLayer;
  begin
    if not CheckCIFARFile() then begin
      //application.Terminate;
      exit;
    end;
    WriteLn('Creating Resize48 Neural Network...');
    NN := TNNet.Create();
    FirstBranch := NN.AddLayer49([
      TNNetInput.Create4(40, 40, 3),                                                                             // Layer 0
      TNNetConvolutionLinear.Create({Features=}64, {FeatureSize=}5, 
                  {Padding=}2, {Stride=}1, {SuppressBias=}1), // Layer 1
      TNNetMaxPool.Create44(4,0,0),                                                                                   // Layer 2
      TNNetMovingStdNormalization.Create(),                                                                     // Layer 3
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1,
                                  {Stride=}1, {SuppressBias=}1),   // Layer 4
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1,
                                  {Stride=}1, {SuppressBias=}1),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1,
                                  {Stride=}1, {SuppressBias=}1),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1,
                                  {Stride=}1, {SuppressBias=}1),  //*)
      TNNetDropout.Create12(0.5,1),
      TNNetMaxPool.Create44(2,0,0),
      TNNetFullConnectLinear.Create28(10,0),
      TNNetSoftMax.Create()
    ]);
    
    //TNNetSwish.Create();

   (* SecondBranch := NN.AddLayerAfter([
      TNNetAvgPool.Create(2),
      TNNetConvolutionSharedWeights.Create(NN.Layers[1]),
      TNNetMaxPool.Create(4),
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionSharedWeights.Create(NN.Layers[4]),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1)
    ], 0);  *)
    
   

  //  NN.DebugStructure();
    CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes, 
                                       ImgTestVolumes, csEncodeRGB);

    ImgTrainingVolumes.ResizeImage(40, 40);
    ImgValidationVolumes.ResizeImage(40, 40);
    ImgTestVolumes.ResizeImage(40, 40);
    
    NeuralFit:= TNeuralImageFit.Create;
    NeuralFit.FileNameBase:= 'SimpleImageClassifierResize48';
    NeuralFit.InitialLearningRate:= 0.001;
    NeuralFit.LearningRateDecay:= 0.01;
    NeuralFit.StaircaseEpochs:= 10;
    NeuralFit.Inertia:= 0.9;
    NeuralFit.L2Decay:= 0.00001;
    NeuralFit.MaxCropSize := 12;
    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes,
                                  {NumClasses=}10, {batchsize=}64, {epochs=}50);
    NeuralFit.Free;

    NN.Free;
    ImgTestVolumes.Free;
    ImgValidationVolumes.Free;
    ImgTrainingVolumes.Free;
    //application.Terminate;
  end;

var
  myApplication: TTestCNNAlgo;
begin
  myApplication := TTestCNNAlgo.Create(nil);
  myApplication.Title:='CIFAR-10 Classification 48 Example';
  //((myApplication.Run;
  //TTestCNNAlgoDoRun();
  myApplication.Free;
end.


ref: Epochs: 50 Examples seen:2000000 Validation Accuracy: 0.8441 Validation Error: 0.4530 Validation Loss: 0.4801 Total time: 3716.52min
Image mX4 FThreadNN[0].DebugWeights(); skipped...
Starting Testing.
Epochs: 50 Examples seen:2000000 Test Accuracy: 0.8418 Test Error: 0.4602 Test Loss: 0.4844 Total time: 3744.44min
Epoch time: 43.0000 minutes. 50 epochs: 36.0000 hours.
Epochs: 50. Working time: 62.41 hours.
CAI maXbox Neural Fit Finished.
 mX4 executed: 18/02/2023 08:54:04  Runtime: 9:35:27.654  Memload: 37% use
PascalScript maXbox4 - RemObjects & SynEdit



Epochs: 2 Examples seen:80000 Validation Accuracy: 0.4160 Validation Error: 1.4152 Validation Loss: 1.6409 Total time: 90.83min
Image mX4 FThreadNN[0].DebugWeights(); skipped...
Epoch time: 26.0000 minutes. 2 epochs: 0.8700 hours.
Epochs: 2. Working time: 1.51 hours.
CAI maXbox Neural Fit Finished.
 mX4 executed: 14/02/2023 17:21:15  Runtime: 1:31:21.560  Memload: 44% use
PascalScript maXbox4 - RemObjects & SynEdit

39040 Examples seen. Accuracy: 0.4527 Error: 1.21989 Loss: 1.37203 Threads: 1 Forward time: 14.00s Backward time: 2.10s Step time: 79.68s
39680 Examples seen. Accuracy: 0.4547 Error: 1.41435 Loss: 1.60288 Threads: 1 Forward time: 13.79s Backward time: 2.18s Step time: 80.38s
40000 of samples have been processed.
Debug SavetokenStructureToString: 
Debug structuretostring: -1)TNNetInput:48;48;3;0;0;0;0;0#0)TNNetConvolutionLinear:64;5;2;1;1;0;0;0#1)TNNetPoolBase:4;4;0;0;0;0;0;0#2)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0#3)TNNetConvolutionReLU:64;3;1;1;1;0;0;0#4)TNNetConvolutionReLU:64;3;1;1;1;0;0;0#5)TNNetConvolutionReLU:64;3;1;1;1;0;0;0#6)TNNetConvolutionReLU:64;3;1;1;1;0;0;0#7)TNNetDropout:2;1;0;0;0;0;0;0#8)TNNetPoolBase:2;2;0;0;0;0;0;0#9)TNNetFullConnectLinear:10;1;1;0;0;0;0;0#10)TNNetSoftMax:0;0;0;0;0;0;0;0
Debug Fit TNNetDataParallelismCloneLen: 3434741
Debug TNNet.Struct.LoadFromString ST: 
-1)TNNetInput:48;48;3;0;0;0;0;0
0)TNNetConvolutionLinear:64;5;2;1;1;0;0;0
1)TNNetPoolBase:4;4;0;0;0;0;0;0
2)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
3)TNNetConvolutionReLU:64;3;1;1;1;0;0;0
4)TNNetConvolutionReLU:64;3;1;1;1;0;0;0
5)TNNetConvolutionReLU:64;3;1;1;1;0;0;0
6)TNNetConvolutionReLU:64;3;1;1;1;0;0;0
7)TNNetDropout:2;1;0;0;0;0;0;0
8)TNNetPoolBase:2;2;0;0;0;0;0;0
9)TNNetFullConnectLinear:10;1;1;0;0;0;0;0
10)TNNetSoftMax:0;0;0;0;0;0;0;0

debug createLayer():TNNetInput-addLayerAfter-1...TNNetInput:48;48;3;0;0;0;0;0
debug createLayer():TNNetConvolutionLinear-addLayerAfter0...TNNetConvolutionLinear:64;5;2;1;1;0;0;0
debug createLayer():TNNetPoolBase-addLayerAfter1...TNNetPoolBase:4;4;0;0;0;0;0;0
debug createLayer():TNNetMovingStdNormalization-addLayerAfter2...TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter3...TNNetConvolutionReLU:64;3;1;1;1;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter4...TNNetConvolutionReLU:64;3;1;1;1;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter5...TNNetConvolutionReLU:64;3;1;1;1;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter6...TNNetConvolutionReLU:64;3;1;1;1;0;0;0
debug createLayer():TNNetDropout-addLayerAfter7...TNNetDropout:2;1;0;0;0;0;0;0
debug createLayer():TNNetPoolBase-addLayerAfter8...TNNetPoolBase:2;2;0;0;0;0;0;0
debug createLayer():TNNetFullConnectLinear-addLayerAfter9...TNNetFullConnectLinear:10;1;1;0;0;0;0;0
debug createLayer():TNNetSoftMax-addLayerAfter10...TNNetSoftMax:0;0;0;0;0;0;0;0
Debug Struct.LoadFromString: 2
Debug TNNet Loaded Data Layers:12  SCount:12
Debug Data.LoadFromString: 2
Debug TNNet.Struct.LoadFromString ST: 
-1)TNNetInput:48;48;3;0;0;0;0;0
0)TNNetConvolutionLinear:64;5;2;1;1;0;0;0
1)TNNetPoolBase:4;4;0;0;0;0;0;0
2)TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
3)TNNetConvolutionReLU:64;3;1;1;1;0;0;0
4)TNNetConvolutionReLU:64;3;1;1;1;0;0;0
5)TNNetConvolutionReLU:64;3;1;1;1;0;0;0
6)TNNetConvolutionReLU:64;3;1;1;1;0;0;0
7)TNNetDropout:2;1;0;0;0;0;0;0
8)TNNetPoolBase:2;2;0;0;0;0;0;0
9)TNNetFullConnectLinear:10;1;1;0;0;0;0;0
10)TNNetSoftMax:0;0;0;0;0;0;0;0

debug createLayer():TNNetInput-addLayerAfter-1...TNNetInput:48;48;3;0;0;0;0;0
debug createLayer():TNNetConvolutionLinear-addLayerAfter0...TNNetConvolutionLinear:64;5;2;1;1;0;0;0
debug createLayer():TNNetPoolBase-addLayerAfter1...TNNetPoolBase:4;4;0;0;0;0;0;0
debug createLayer():TNNetMovingStdNormalization-addLayerAfter2...TNNetMovingStdNormalization:0;0;0;0;0;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter3...TNNetConvolutionReLU:64;3;1;1;1;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter4...TNNetConvolutionReLU:64;3;1;1;1;0;0;0
debug createLayer():TNNetConvolutionReLU-addLayerAfter5...TNNetConvolutionReLU:64;3;1;1;1;0;0;0
Exception: Out of memory.


37120 Examples seen. Accuracy: 0.4568 Error: 1.29219 Loss: 1.68322 Threads: 1 Forward time: 12.13s Backward time: 2.01s Step time: 66.59s
37760 Examples seen. Accuracy: 0.4562 Error: 1.40706 Loss: 1.61520 Threads: 1 Forward time: 12.01s Backward time: 2.10s Step time: 65.83s
38400 Examples seen. Accuracy: 0.4591 Error: 1.28368 Loss: 1.30424 Threads: 1 Forward time: 12.07s Backward time: 2.04s Step time: 65.75s
39040 Examples seen. Accuracy: 0.4584 Error: 1.41554 Loss: 1.60082 Threads: 1 Forward time: 10.17s Backward time: 1.62s Step time: 66.57s
39680 Examples seen. Accuracy: 0.4617 Error: 1.30342 Loss: 1.61960 Threads: 1 Forward time: 12.16s Backward time: 1.95s Step time: 67.56s
