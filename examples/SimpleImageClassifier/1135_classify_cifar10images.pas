unit classify_cifar10_images;

//{$mode objfpc}{$H+}
//http://www.softwareschule.ch/examples/uPSI_NeuralNetworkCAI.txt
//https://github.com/mabudrais/CAI-NEURAL-API-Test
//https://github.com/maxkleiner/neural-api/blob/master/neural/neuraldatasets.pas
//https://github.com/maxkleiner/neural-api/blob/master/neural/neuralvolume.pas
//https://t.co/r6QE2kaEuP
{the rpblem(bug) is it predicts always a cat - we work on it}

interface

{uses
  Classes, SysUtils, Forms, Controls, Graphics, Dialogs, StdCtrls, ExtCtrls, Grids,
  neuralnetwork, neuralvolume, neuralfit, neuraldatasets, neuralopencl,
  neuralvolumev, FileUtil;   test from laz to mX4}  

type

  { TForm1 }

  TForm1 = {class(}TForm;
  var
    Button1: TButton;
    ComboBox1: TComboBox;
    Image1: TImage;
    Label1: TLabel;
    StringGrid1: TStringGrid;
    procedure TForm1Button1Click(Sender: TObject);
    procedure TForm1ComboBox1Change(Sender: TObject);
    procedure TForm1FormCreate(Sender: TObject);
  //private

  //public
  
 Const PICPATH = 'C:\maXbox\EKON_BASTA\EKON24\examples\';  
       TRAINPATH = 'C:\maXbox\EKON_BASTA\EKON24\examples\';

  //end;

var
  Form1: TForm1;

implementation

//{$R *.lfm}

{ TForm1 }

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

{ ref: csTinyImageLabel: array[0..9] of string =
    (
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
    );  }

procedure SetColumnFullWidth(Grid: TStringGrid; ACol: Integer);
var I: Integer;
    FixedWidth: Integer;
begin
  with Grid do
    if ACol >= FixedCols then begin
      FixedWidth:= 0;
      for I:= 0 to FixedCols - 1 do
        Inc2(FixedWidth, ColWidths[I] + GridLineWidth);
      ColWidths[ACol]:= form1.ClientWidth- FixedWidth- GridLineWidth;     
    end;
end;

const
  MIN_COL_WIDTH = 15;
procedure AutoSizeGridColumns(Grid: TStringGrid);
var Col, row : Integer;
    ColWidth, CellWidth: Integer;
begin
  Grid.Canvas.Font.Assign(Grid.Font);
  for Col:= 0 to Grid.ColCount -1 do begin
    ColWidth:= Grid.Canvas.TextWidth(Grid.Cells[Col, 0]);
    for Row:= 0 to Grid.RowCount - 1 do begin
      CellWidth:= Grid.Canvas.TextWidth(Grid.Cells[Col, Row]);
      if CellWidth > ColWidth then
        ColWidth:= CellWidth
    end;
    Grid.ColWidths[Col]:= ColWidth + MIN_COL_WIDTH;
  end;
end;

procedure TForm1Button1Click(Sender: TObject);
var
  NN: THistoricalNets; //TNNet; //THistoricalNets;
  pOutPut, pInput: TNNetVolume;
  k: integer;

begin
  NN:= THistoricalNets.create; //TNNet.Create();
  NN.LoadFromFile(TRAINPATH+'SimpleImageClassifierEkon26_70.nn');
  //NN.EnableDropouts(false);
  pInput := TNNetVolume.Create0(32, 32, 3, 1);
  pOutPut := TNNetVolume.Create0(10, 1, 1, 1);
  writeln('picname: '+extractfilename(ComboBox1.text));
  writeln('picsize1 '+Format('%d×%d',[image1.picture.Width,image1.picture.Height]));
  //LoadPictureIntoVolume(Image1.picture, pInput);
  //image1.picture.loadfromfile(ComboBox1.text);
  //writeln(image1.picture.tag)
  LoadImageIntoVolume(image1, pInput); 
  pInput.RgbImgToNeuronalInput(csEncodeRGB);
  //pInput.NeuronalInputToRgbImg(csEncodeRGB);           
  //LoadImageFromFileIntoVolume('C:\maXbox\EKON_BASTA\EKON24\examples\csautomobile4.png',
    //                                           pinput)
  //loadCifar10Dataset2(ImgVolumes, 5, 0,csEncodeRGB);
  // pInput.RgbImgToNeuronalInput(csEncodeRGB);
  writeln('picsize2 '+itoa(pinput.size));
  NN.Compute65(pInput,0);
  NN.GetOutput(pOutPut);
  writeln(itoa(pOutPut.GetClass()));   
  Label1.Caption:= cs10Labels[pOutPut.GetClass()];
  for k := 1 to 10 do
    StringGrid1.Cells[1,k]:= 
                    FloatToStr(NN.Layers[NN.Layers.Count-2].Output.Raw[k-1]);
  NN.Free;
  //pinput.clear;
  pInput.Free;
  pOutPut.Free;
end;

procedure TForm1ComboBox1Change(Sender: TObject);
begin
  if FileExists(ComboBox1.text) then begin
    Image1.Picture.LoadFromFile(ComboBox1.text);
  label1.Caption:= extractfilename(ComboBox1.text); 
 end; 
end;

procedure TForm1FormCreate(Sender: TObject);
var k: integer;
begin
  for k := 0 to 9 do
    StringGrid1.Cells[0, k+1] := cs10Labels[k];
  //FindAllFiles(ComboBox1.Items, 'csdata');
  if ComboBox1.Items.Count > 0 then begin
    ComboBox1.text := ComboBox1.Items[0];
    if FileExists(ComboBox1.text) then
      Image1.Picture.LoadFromFile(ComboBox1.text);
   label1.Caption:= extractfilename(ComboBox1.text);    
  end;
end;

procedure loadAIForm;
begin
Form1:= TForm1.create(self);
 with form1 do begin
  Left := 486
  Height:= 422
  Top:= 145
  Width:= 413
  Caption:= 'Form1 Classify'
  ClientHeight:= 422
  ClientWidth:= 413
  icon.loadfromresourcename(hinstance, 'XLV');
  OnCreate:= @TForm1formCreate;
  //LCLVersion:= '2.0.4.0'
  Button1:= TButton.create(form1)
  show;
  with button1 do begin
    //AnchorSideLeft.Control:= Owner
    //AnchorSideTop.Control:= ComboBox1
    //AnchorSideTop.Side:= asrBottom
    parent:= form1;
    Left:= 5
    Height:= 25
    Top:= 60
    Width:= 75
    //BorderSpacing.Left:= 2
    //BorderSpacing.Top:= 2
    Caption:= 'classify'
    OnClick:= @TForm1Button1Click;
    TabOrder:= 0
  end;
  Image1:= TImage.create(form1)
  with image1 do begin
    //AnchorSideLeft.Control:= Owner
    //AnchorSideTop.Control:= Owner
    parent:= form1;
    Left:= 5
    Height:= 32
    Top:= 2
    Width:= 32
    //BorderSpacing.Left:= 2
    //BorderSpacing.Top:= 2
  end;
  // label1 as target label class!
  Label1:= TLabel.create(form1)
  with label1 do begin
    //AnchorSideLeft.Control:= Image1
    //AnchorSideLeft.Side:= asrBottom
    //AnchorSideTop.Control:= Image1
    //AnchorSideTop.Side:= asrCenter
    parent:= form1;
    Left:= 50
    Height:= 13
    Top:= 12
    Width:= 31
    //BorderSpacing.Left:= 10
    Caption:= 'Label1'
    ParentColor:= False
  end;
  StringGrid1:= TStringGrid.create(form1)
  with stringgrid1 do begin
    //AnchorSideLeft.Control:= Owner
    //AnchorSideTop.Control:= Button1
    //AnchorSideTop.Side:= asrBottom
    parent:= form1;
    Left:= 5
    Height:= 280;
    Top:= 87
    Width:= 193
    //BorderSpacing.Left:= 2
    //BorderSpacing.Top:= 2
    ColCount:= 2
    RowCount:= 11
    TabOrder:= 1
    ColWidths[1]:= 124;
    Cells[0,0]:= 'type';
    Cells[1,0]:= ' [0,1]';
   {    1 0  0  'type'   }
    //)
  end;
  ComboBox1:= TComboBox.create(form1)
  with combobox1 do begin
    //AnchorSideLeft.Control:= Owner
    //AnchorSideTop.Control:= Image1
    //AnchorSideTop.Side:= asrBottom
    parent:= form1;
    Left:= 5
    Height:= 21
    Hint:= 'choose an image'
    Top:= 37
    Width:= 376
    //BorderSpacing.Left:= 2
    //BorderSpacing.Top:= 3
    items.add(PICPATH+'csbird9.bmp');
    items.add(PICPATH+'csdog1.bmp');
    items.add(PICPATH+'csautomobile4.bmp');
    items.add(PICPATH+'csairplane4.bmp');
    items.add(PICPATH+'csairplane1.bmp');
    ItemHeight:= 13
    OnChange:= @TForm1ComboBox1Change;
    ParentShowHint:= False
    ShowHint:= True
    TabOrder:= 2
    Text:= 'ComboBox1'
  end;
   TForm1FormCreate(self);
 end ; //form1
end;

begin //@main

  setClassifierLabels;
  loadAIForm();

 end.
End.

ref:   Compute - Wrong Input Size:0 Expected size is:3072 Have you missed the TNNetInput layer?    = 32*32*3
