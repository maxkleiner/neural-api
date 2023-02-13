program Hypotenuse_mX4;
(*     not really works - got an exception with callback function
   RegisterProperty('InferHitFn', 'TNNetInferHitFn', iptrw);
       NFit.InferHitFn := Nil; //@LocalFloatCompare;

Hypotenuse: learns how to calculate hypotenuse sqrt(X^2 + Y^2).
Copyright (C) 2019 Joao Paulo Schwarz Schuler

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
*)

//{$mode objfpc}{$H+}

(*
uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes,
  neuralnetwork,
  neuralvolume,
  neuralfit;  *)

  function CreateHypotenusePairList(MaxCnt: integer): TNNetVolumePairList;
  var
    Cnt: integer;
    LocalX, LocalY, Hypotenuse: TNeuralFloat;
  begin
    Result := TNNetVolumePairList.Create(true);
    for Cnt := 1 to MaxCnt do begin
      LocalX := Random(100);
      LocalY := Random(100);
      Hypotenuse := sqrt(LocalX*LocalX + LocalY*LocalY);

      Result.Add(
        TNNetVolumePair.Create82(
          TNNetVolume.Create1([LocalX, LocalY]),
          TNNetVolume.Create1([Hypotenuse])
        )
      );
    end;
  end;

  // Returns TRUE if difference is smaller than 0.1 .
  function LocalFloatCompare(A, B: TNNetVolume; ThreadId: integer): boolean;
  var FDataA, FDataB: TNeuralFloatArray;
  begin
   FDataA:= A.FData;     
    FDataB:= B.FData;  
    //Result := ( Abs(A.FData[0]-B.FData[0])<0.1 );
    Result := ( Abs(FDataA[0]-FDataB[0])<0.1);
  end;

  procedure RunAlgo();
  var
    NN: TNNet;
    NFit: TNeuralFit;
    TrainingPairs, ValidationPairs, TestPairs: TNNetVolumePairList;
    Cnt: integer;
    pOutPut: TNNetVolume;   FDataA, FDataB: TNeuralFloatArray;
  begin
    NN := TNNet.Create();
    NFit := TNeuralFit.Create();
    TrainingPairs := CreateHypotenusePairList(10000);
    ValidationPairs := CreateHypotenusePairList(1000);
    TestPairs := CreateHypotenusePairList(1000);

    NN.AddLayer2([
      TNNetInput.Create3(2),
      TNNetFullConnectReLU.Create30(32,0),
      TNNetFullConnectReLU.Create30(32,0),
      TNNetFullConnectLinear.Create28(1,0)
    ]);

    WriteLn('Hypotenuse Computing...');
    NFit.InitialLearningRate := 0.00001;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    //NFit.InferHitFn := Nil; //@LocalFloatCompare;
    //MonopolarCompare( A, B: TNNetVolume; ThreadId : integer) : boolean');
    //bipolarCompare(pOutPut, poutput, 1) ;
    
    NFit.Fit(NN, TrainingPairs, ValidationPairs, TestPairs, 
                                          {batchsize=}32, {epochs=}50);
    NN.DebugWeights();

    pOutPut:= TNNetVolume.Create0({pSizeX=}1, {pSizeY=}1, {pDepth=}1, {FillValue=}1);
    writeln('bipolarCompare: '+botostr(bipolarCompare(pOutPut, poutput, 0))) ;

    // tests the learning
    for Cnt := 0 to 9 do begin
      NN.Compute65(TestPairs[Cnt].I,0);
      NN.GetOutput(pOutPut);
      FDATAA:= TestPairs[Cnt].I.FData;   
      WriteLn
      ( 'Inputs:'+
        //TestPairs[Cnt].I.FData[0]:5:2,', ',
        //TestPairs[Cnt].I.FData[1]:5:2,' - ',
        flots(FDATAA[0])+', '+
        flots(FDATAA[1])+' - '+
        'Output:'+
        //pOutPut.Raw[0]:5:2,' ',
        format('%.3f ',[pOutPut.Raw[0]])+' '+
        ' Desired Output:'+
        //TestPairs[Cnt].O.FData[0]:5:2
         flots(FDATAA[0])
      );
    end;
    pOutPut.Free;
    TestPairs.Free;
    ValidationPairs.Free;
    TrainingPairs.Free;
    NFit.Free;
    NN.Free;
    Write('Press ENTER simulate to exit.');
    //7/ReadLn;
  end;

//var
  // Stops Lazarus errors
  //maxform1.Application: record Title:string; end;

begin
  Application.Title:='Hypotenuse Example';
  RunAlgo();
end.
