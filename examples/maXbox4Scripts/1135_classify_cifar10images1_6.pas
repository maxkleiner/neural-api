unit classify_cifar10_images2lazTutor42_1_6;

//{$mode objfpc}{$H+}       resize to 40*40 top model
//http://www.softwareschule.ch/examples/uPSI_NeuralNetworkCAI.txt
//https://github.com/mabudrais/CAI-NEURAL-API-Test
//https://github.com/maxkleiner/neural-api/blob/master/neural/neuraldatasets.pas
//https://github.com/maxkleiner/neural-api/blob/master/neural/neuralvolume.pas
//http://www.softwareschule.ch/examples/uPSI_neuralvolume.txt
//https://t.co/r6QE2kaEuP                                 
{the rpblem(bug) is it predicts always a cat - solved!, with findfiles}
//Tutor and Demo 105 for Lazarus CAI Package Integration

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
    chkboxdrop: TCheckbox;
    Image1, image2: TImage;
    Label1, label2, lbldropout: TLabel;
    StringGrid1: TStringGrid;
    procedure TForm1Button1Click(Sender: TObject);
    procedure TForm1ComboBox1Change(Sender: TObject);
    procedure TForm1FormCreate(Sender: TObject);
  //private

  //public
  
 Const PICPATH = '.\data\';  
       TRAINPATH = '.\model\ClassifyCNNResize40_84.nn';
                            
var
  Form1: TForm1;

implementation

//{$R *.lfm}
{ TForm1 }

var cs10Labels: array[0..9] of string;
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
    ( 'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'  );  }

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
var Col, row: Integer;
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

{TTinyImage = packed record
    bLabel: byte;
    R, G, B: TTinyImageChannel;
  end;}

procedure TForm1Button1Click(Sender: TObject);
var
  NN: THistoricalNets; //TNNet; 
  pOutPut, pInput: TNNetVolume; TI: TTinyImage;
  k: integer;
begin
  NN:= THistoricalNets.create; //TNNet.Create();
  //NN.LoadFromFile(TRAINPATH+'SimpleImageClassifierEkon26_70.nn');
  NN.LoadFromFile(TRAINPATH);
  label2.caption:= 'load: '+TRAINPATH;
     if chkboxdrop.checked then
       NN.EnableDropouts(true) else
       NN.EnableDropouts(false); 
  pInput:= TNNetVolume.Create0(40, 40, 3, 1);
  pOutPut:= TNNetVolume.Create0(10, 1, 1, 1);
  // pinput.ReSize(40,40,3);
  //image1.picture.width:= 40;
  writeln('picname: '+extractfilename(ComboBox1.text));
  writeln('picsize1 '+Format('%d×%d',[image1.picture.Width,image1.picture.Height]));
  LoadPictureIntoVolume(image1.picture, pinput); 
  pinput.ReSize(40,40,3);
  //loadCifar10Dataset2(ImgVolumes,5,0,csEncodeRGB);
  pInput.RgbImgToNeuronalInput(csEncodeRGB);
  writeln('picsize2 '+itoa(pinput.size));
  NN.Compute65(pInput,0);
  NN.GetOutput(pOutPut);
  writeln('resget class type: '+itoa(pOutPut.GetClass()));
  stringgrid1.selection:= 
                  TGridRect(Rect(0,pOutPut.GetClass+1,0,pOutPut.GetClass+1));    
  Label1.Caption:= 'predicts: '+cs10Labels[pOutPut.GetClass()];
  for k:= 1 to 10 do
    StringGrid1.Cells[1,k]:= 
                    FloatToStr(NN.Layers[NN.Layers.Count-2].Output.Raw[k-1]);
  NN.Free;
  pInput.Free;
  pOutPut.Free;
end;

procedure TForm1ComboBox1Change(Sender: TObject);
begin
  if FileExists(ComboBox1.text) then begin
    Image1.Picture.LoadFromFile(ComboBox1.text);
    Image2.Picture.LoadFromFile(ComboBox1.text);
    label1.Caption:= extractfilename(ComboBox1.text); 
  end; 
end;

procedure TForm1FormCreate(Sender: TObject);
var k,t: integer;
  items : TStringList;
begin
  items := TStringList.create;
  for k := 0 to 9 do
    StringGrid1.Cells[0, k+1] := cs10Labels[k];
  //FindAllFiles(ComboBox1.Items, 'csdata');
  FindFiles(exepath+'data', '*.bmp',items);
  writeln(items.text);
  for t:= 1 to items.count-1 do  
     ComboBox1.Items.add(items[t]);
  if ComboBox1.Items.Count > 0 then begin
    ComboBox1.text:= ComboBox1.Items[0];
    if FileExists(ComboBox1.text) then begin
      Image1.Picture.LoadFromFile(ComboBox1.text);
      Image2.Picture.LoadFromFile(ComboBox1.text);
      label1.Caption:= extractfilename(ComboBox1.text); 
    end;     
  end;
end;

procedure loadAIForm;
begin
Form1:= TForm1.create(self);
 with form1 do begin
  setbounds(145, 486, 430, 440)
  Caption:= 'Form1 maXbox CAI_Classify 1.6'
  ClientHeight:= 430; ClientWidth:= 425;
  icon.loadfromresourcename(hinstance, 'XLV');
  OnCreate:= @TForm1formCreate;
  //LCLVersion:= '2.0.4.0'
  Button1:= TButton.create(form1)
  show;
  with button1 do begin
    parent:= form1;
    Left:= 5; Height:= 28
    Top:= 106; Width:= 100
    Caption:= '&Classify'
    OnClick:= @TForm1Button1Click;
    TabOrder:= 0
  end;
  Image1:= TImage.create(form1)
  with image1 do begin
    //AnchorSideLeft.Control:= Owner
    parent:= form1;
    Top:= 5; Left:= 5
    Height:= 40  //32
    Width:= 40   //32
    stretch:= true;
  end;
  Image2:= TImage.create(form1)
  with image2 do begin
    parent:= form1;
    Top:= 5; Left:= 150
    Height:= 64  //32
    Width:= 64   //32
    stretch:= true;
  end;
  // label1 as target label class!
  Label1:= TLabel.create(form1)
  with label1 do begin
    parent:= form1;
    Left:= 50; height:= 13
    Top:= 12;  Width:= 31
    //BorderSpacing.Left:= 10
    Caption:= 'Label1'
    ParentColor:= False
  end;
  Label2:= TLabel.create(form1)
  with label2 do begin
    parent:= form1;
    Left:= 220; height:= 13
    Top:= 12;  Width:= 31
    //BorderSpacing.Left:= 10
    Caption:= 'model'
    ParentColor:= False
  end;
  Lbldropout:= TLabel.create(form1)
  with lbldropout do begin
    parent:= form1;
    Left:= 220; height:= 13
    Top:= 50;  Width:= 31               
    Caption:= 'dropout:'
    ParentColor:= False
  end;
  chkboxdrop:= TCheckbox.create(form1)
  with chkboxdrop do begin
    parent:= form1;
    Left:= 270; height:= 13
    Top:= 52;  Width:= 31 
    checked:= true;              
   // Caption:= 'dropout'
    ParentColor:= False
  end;
  StringGrid1:= TStringGrid.create(form1)
  with stringgrid1 do begin
    parent:= form1;
    Left:= 5; Height:= 279;
    Top:= 142; width:= 193
    ColCount:= 2
    RowCount:= 11
    TabOrder:= 1
    ColWidths[1]:= 124;
    Cells[0,0]:= 'type';
    Cells[1,0]:= 'probability +-[-60,90]';
   {    1 0  0  'type'   } //)
  end;
  ComboBox1:= TComboBox.create(form1)
  with combobox1 do begin
    parent:= form1;
    Left:= 5; Height:= 21
    Hint:= 'choose an image'
    Top:= 77
    Width:= 376
    items.add(PICPATH+'ship140.bmp');
    items.add(PICPATH+'bird940.bmp');
    items.add(PICPATH+'dog140.bmp');
    items.add(PICPATH+'automobile440.bmp');
    items.add(PICPATH+'airplane4.bmp');
    items.add(PICPATH+'airplane1.bmp');
    items.add(PICPATH+'frog4.bmp');
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
  //TestTNNetVolume();
  setClassifierLabels;
  loadAIForm();
 end.
End.

ref: https://www.cs.toronto.edu/~kriz/cifar.html  
     https://github.com/mabudrais/CAI-NEURAL-API-Test/blob/master/unit1.pas
Compute - Wrong Input Size:0 Expected size is:3072 Have you missed the TNNetInput layer?    = 32*32*3

Lazarus Unit laZ 2.0.8

C:\Program Files\Streaming\maxbox4\maxbox47590\maxbox4\lib\cai-svncode-r655-trunk-lazarus-libs\cai-svncode-r1014-trunk-lazarus-neural
C:\Program Files\Streaming\maxbox4\maxbox47590\maxbox4\lib\lazarus-ccr-svn-3722\lazarus-ccr-svn-3722\components\multithreadprocs
C:\Program Files\Streaming\maxbox4\maxbox47590\maxbox4\resized\neural-api-master\neural-api-master\neural
C:\Programme\Streaming\lazarus\lcl\units\x86_64-win64\win32
C:\Programme\Streaming\lazarus\lcl\units\x86_64-win64
C:\Programme\Streaming\lazarus\components\lazutils\lib\x86_64-win64
C:\Programme\Streaming\lazarus\packager\units\x86_64-win64
C:\Program Files\Streaming\maxbox4\maxbox47590\maxbox4\resized\CAI-NEURAL-API-Test-master\CAI-NEURAL-API-Test-master

program classsifyimage;

//{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Interfaces, // this includes the LCL widgetset
  Forms,Unit1
  { you can add units after this };

//{$R *.res}

begin
  RequireDerivedFormResource:=True;
  Application.Scaled:=True;
  Application.Initialize;
  Application.CreateForm(TForm1,Form1);
  Application.Run;
end.

unit Unit1;

//{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, Graphics, Dialogs, StdCtrls, ExtCtrls, Grids,
  neuralnetwork, neuralvolume, neuralfit, neuraldatasets, neuralopencl,
  neuralvolumev, FileUtil;

type

  { TForm1 }
  TForm1 = class(TForm)
    Button1: TButton;
    ComboBox1: TComboBox;
    Image1: TImage;
    Label1: TLabel;
    StringGrid1: TStringGrid;
    procedure Button1Click(Sender: TObject);
    procedure ComboBox1Change(Sender: TObject);
    procedure FormCreate(Sender: TObject);
  private
  public

  end;

var
  Form1: TForm1;

implementation

//{$R *.lfm}

{ TForm1 }

procedure TForm1.Button1Click(Sender: TObject);
var NN: THistoricalNets;
  pOutPut, pInput: TNNetVolume;
  k: integer;
begin
  NN:= THistoricalNets.Create();
  NN.LoadFromFile('SimpleImageClassifier.nn');
  pInput := TNNetVolume.Create(32, 32, 3, 1);
  pOutPut := TNNetVolume.Create(10, 1, 1, 1);
  LoadPictureIntoVolume(Image1.Picture, pInput);
  // pInput.RgbImgToNeuronalInput(csEncodeRGB);
  NN.Compute(pInput);
  NN.GetOutput(pOutPut);
  Label1.Caption := csTinyImageLabel[pOutPut.GetClass()];
  for k := 1 to 10 do
    StringGrid1.Cells[1,k]:= FloatToStr(NN.Layers[NN.Layers.Count-2].Output.Raw[k-1]);
  NN.Free;
  pInput.Free;
  pOutPut.Free;
end;

procedure TForm1.ComboBox1Change(Sender: TObject);
begin
  if FileExists(ComboBox1.Caption) then
    Image1.Picture.LoadFromFile(ComboBox1.Caption);
end;

procedure TForm1.FormCreate(Sender: TObject);
var k: integer;
begin
  for k := 0 to 9 do
    StringGrid1.Cells[0, k+1] := csTinyImageLabel[k];
  FindAllFiles(ComboBox1.Items, 'data');
  if ComboBox1.Items.Count > 0 then begin
    ComboBox1.Caption := ComboBox1.Items[0];
    if FileExists(ComboBox1.Caption) then
      Image1.Picture.LoadFromFile(ComboBox1.Caption);
  end;
end;

end.
