program SimpleImageClassifierSharedWeights_mX4;
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
    WriteLn('Creating SharedWeights Neural Network...');
    NN := TNNet.Create();
    FirstBranch := NN.AddLayer49([
      TNNetInput.Create4(32, 32, 3),                                                                             // Layer 0
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
                                  {Stride=}1, {SuppressBias=}1),
      TNNetDropout.Create12(0.5,1),
      TNNetMaxPool.Create44(2,0,0)
    ]);

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
    
    SecondBranch := NN.AddLayer49([
      TNNetAvgPool.Create45(2),
      TNNetConvolutionSharedWeights.Create39(NN.Layers[1]),
      TNNetMaxPool.Create44(4,0,0),
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionSharedWeights.Create39(NN.Layers[4]),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1,
                                  {Stride=}1, {SuppressBias=}1),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1,
                                  {Stride=}1, {SuppressBias=}1),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1,
                                  {Stride=}1, {SuppressBias=}1)
       ]);  //*)

    NN.AddLayer49([
      TNNetDeepConcat.Create20([FirstBranch, SecondBranch]),
      TNNetFullConnectLinear.Create28(10,0),
      TNNetSoftMax.Create()
    ]);

    NN.DebugStructure();
    CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes, 
                                       ImgTestVolumes, csEncodeRGB);

    NeuralFit:= TNeuralImageFit.Create;
    NeuralFit.FileNameBase:= 'SimpleImageClassifierDeepConcat';
    NeuralFit.InitialLearningRate:= 0.001;
    NeuralFit.LearningRateDecay:= 0.01;
    NeuralFit.StaircaseEpochs:= 10;
    NeuralFit.Inertia:= 0.9;
    NeuralFit.L2Decay:= 0.00001;
    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes,
                                  {NumClasses=}10, {batchsize=}64, {epochs=}50);
    NeuralFit.Free;

    NN.Free;
    ImgTestVolumes.Free;
    ImgValidationVolumes.Free;
    ImgTrainingVolumes.Free;
    application.Terminate;
  end;

var
  myApplication: TTestCNNAlgo;
begin
  myApplication := TTestCNNAlgo.Create(nil);
  myApplication.Title:='CIFAR-10 Classification Example';
  //((myApplication.Run;
  //TTestCNNAlgo;
  myApplication.Free;
end.
