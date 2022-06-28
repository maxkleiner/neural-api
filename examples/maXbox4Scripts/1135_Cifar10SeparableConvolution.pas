//program Cifar10Resize_mX4_2;
(*
 Coded by Joao Paulo Schwarz Schuler.
 https://github.com/joaopauloschuler/neural-api
 https://github.com/maxkleiner/neural-api/blob/master/examples/Cifar10Resize/Cifar10Resize.lpr
 https://github.com/maxkleiner/neural-api/blob/master/neural/neuralthread.pas
 https://colab.research.google.com/drive/1bKUi4PsVJMZ4rzFxoK7HFJsuCJI6yuXW#scrollTo=4HlQx4wqtXDz
*)
//{$mode objfpc}{$H+}

(*
uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, CustApp, neuralnetwork, neuralvolume,
  Math, neuraldatasets, neuralfit, neuralthread, usuperresolutionexample;

  *)
  
program SeparableConvolution_mX4;
(*
 Coded by Joao Paulo Schwarz Schuler.
 https://github.com/joaopauloschuler/neural-api
*)
//{$mode objfpc}{$H+}

(*uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, CustApp, neuralnetwork, neuralvolume, Math, neuraldatasets, neuralfit;
  *)
  
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
  begin
    if not CheckCIFARFile() then begin
      //Terminate;
      //exit;
        writeln('___terminate & exit___');
    end;

    WriteLn('Creating SeparableConvolution Neural Network...');
    NN := TNNet.Create();
    NN.AddLayer2([
      TNNetInput.Create4(32, 32, 3),
      TNNetConvolutionLinear.Create({NumFeatures=}16 {64}, {featureSize=}5, 
                  {padding=}2, {stride=}1, {SuppressBias=}0).InitBasicPatterns(),
      TNNetMaxPool.Create44(4,0,0)
    ]);
    NN.AddSeparableConvReLU({NumFeatures=}32,{featureSize=}3,{padding=}1,{stride=}1,{DepthMultiplier=}1,{SuppressBias=}0, {pAfterLayer}0);
    NN.AddSeparableConvReLU({NumFeatures=}64,{featureSize=}3,{padding=}1,{stride=}1,{DepthMultiplier=}1,{SuppressBias=}0,0);
   // NN.AddSeparableConvReLU({NumFeatures=}64,{featureSize=}3,{padding=}1,{stride=}1,{DepthMultiplier=}1,{SuppressBias=}0,0);
  //  NN.AddSeparableConvReLU({NumFeatures=}64,{featureSize=}3,{padding=}1,{stride=}1,{DepthMultiplier=}1,{SuppressBias=}0,0);
    NN.AddLayer2([
      TNNetDropout.Create12(0.5,1),
      TNNetMaxPool.Create44(2,0,0),
      TNNetFullConnectLinear.Create28(10,0),
      TNNetSoftMax.Create()
    ]);
    NN.DebugWeights();
    NN.DebugStructure();
    WriteLn('Layers: '+itoa( NN.CountLayers()));
    WriteLn('Neurons: '+itoa( NN.CountNeurons()));
    WriteLn('Weights: '+itoa( NN.CountWeights()));

    CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes,
                                          csEncodeRGB);

    NeuralFit := TNeuralImageFit.Create;
    NeuralFit.FileNameBase := 'SimpleSeparableImageClassifier12';
    NeuralFit.InitialLearningRate := 0.001;  //0.001
    NeuralFit.LearningRateDecay := 0.1 //0.01;
    NeuralFit.StaircaseEpochs := 10;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0.0001;  //0.00001;
    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, {NumClasses=}10, {batchsize=}128, {epochs=}10  {50});
    NeuralFit.Free;

    NN.Free;
    ImgTestVolumes.Free;
    ImgValidationVolumes.Free;
    ImgTrainingVolumes.Free;
    //Terminate;
    writeln('terminate__')
  end;

var
  Application2: TTestCNNAlgo;
begin //@main
  Application2 := TTestCNNAlgo.Create(nil);
  Application2.Title:='CIFAR-10 Separable Convolutions Example';
  //Application2.Run;
  TTestCNNAlgoDoRun;
  Application2.Free;
End.

Doc:   https://raw.githubusercontent.com/maxkleiner/neural-api/master/examples/SeparableConvolution/SeparableConvolution.lpr

https://sourceforge.net/p/cai/svncode/1313/tree/trunk/lazarus/examples/SuperResolution/usuperresolutionexample.pas#l19

ref: 0;0;0;0#10)TNNetDropout:2;1;0;0;0;0;0;0#11)TNNetMaxPool:2;2;0;0;0;0;0;0#12)TNNetFullConnectLinear:10;1;1;0;0;0;0;0#13)TNNetSoftMax:0;0;0;0;0;0;0;0
Epochs: 2 Examples seen:80000 Validation Accuracy: 0.4670 Validation Error: 1.3207 Validation Loss: 1.4977 Total time: 51.62min
Image mX4 FThreadNN[0].DebugWeights(); skipped...
Epoch time: 14.0000 minutes. 2 epochs: 0.4700 hours.
Epochs: 2. Working time: 0.86 hours.
CAI maXbox Neural Fit Finished.
Epochs: 2 Examples seen:80000 Validation Accuracy: 0.4658 Validation Error: 1.3272 Validation Loss: 1.4792 Total time: 17.47min
Image mX4 FThreadNN[0].DebugWeights(); skipped...
Epoch time: 5.9000 minutes. 2 epochs: 0.2000 hours.
Epochs: 2. Working time: 0.29 hours.
CAI maXbox Neural Fit Finished.
Epochs: 1 Examples seen:40000 Validation Accuracy: 0.4659 Validation Error: 1.2762 Validation Loss: 1.4974 Total time: 7.85min
Image mX4 FThreadNN[0].DebugWeights(); skipped...
Epoch time: 4.8000 minutes. 1 epochs: 0.0800 hours.
Epochs: 1. Working time: 0.13 hours.
CAI maXbox Neural Fit Finished.
Image mX4 FThreadNN[0].DebugWeights(); skipped...
Epoch time: 4.9000 minutes. 1 epochs: 0.0820 hours.
Epochs: 1. Working time: 0.13 hours.
Epochs: 1 Examples seen:40000 Validation Accuracy: 0.4650 Validation Error: 1.3208 Validation Loss: 1.5201 Total time: 7.74min
Image mX4 FThreadNN[0].DebugWeights(); skipped...
Epoch time: 4.7000 minutes. 1 epochs: 0.0790 hours.
Epochs: 1. Working time: 0.13 hours.
CAI maXbox Neural Fit Finished.
Epochs: 1 Examples seen:40000 Validation Accuracy: 0.4838 Validation Error: 1.3297 Validation Loss: 1.4652 Total time: 6.74min
Image mX4 FThreadNN[0].DebugWeights(); skipped...
Epoch time: 4.0000 minutes. 1 epochs: 0.0670 hours.
Epochs: 1. Working time: 0.11 hours.
CAI maXbox Neural Fit Finished.
Epochs: 10 Examples seen:400000 Validation Accuracy: 0.6468 Validation Error: 1.0159 Validation Loss: 1.0283 Total time: 77.44min
Image mX4 FThreadNN[0].DebugWeights(); skipped...
Starting Testing.
Epochs: 10 Examples seen:400000 Test Accuracy: 0.6437 Test Error: 1.0146 Test Loss: 1.0337 Total time: 79.84min
Epoch time: 3.6000 minutes. 10 epochs: 0.5900 hours.
Epochs: 10. Working time: 1.33 hours.
CAI maXbox Neural Fit Finished.
terminate__
 mX4 executed: 28/06/2022 21:52:09  Runtime: 1:19:57.803  Memload: 40% use

