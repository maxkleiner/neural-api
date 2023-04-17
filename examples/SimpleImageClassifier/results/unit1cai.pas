unit unit1cai;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, Graphics, Dialogs, StdCtrls, ExtCtrls, Grids,
  neuralnetwork, neuralvolume, neuralfit, neuraldatasets, neuralopencl,
  neuralvolumev, FileUtil;

type

  { TForm1 }

  TForm1 = class(TForm)
    Button1: TButton;
    CheckBox1: TCheckBox;
    CheckBox2: TCheckBox;
    ComboBox1: TComboBox;
    Edit1: TEdit;
    Image1: TImage;
    Label1: TLabel;
    Label2: TLabel;
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

{$R *.lfm}

{ TForm1 }

procedure TForm1.Button1Click(Sender: TObject);
var
  NN: THistoricalNets;
  pOutPut, pInput: TNNetVolume;
  k: integer;

begin
  NN := THistoricalNets.Create();
  //NN.LoadFromFile('.\model\ClassifyCNNModel_70.nn');
  //NN.LoadFromFile('.\ClassifyCNNResize40_84.nn');
   NN.LoadFromFile(edit1.Text);  //'.\model\ClassifyCNNModel_70.nn');
  label2.Caption:= edit1.text; // 'load .\model\ClassifyCNNModel_70.nn';
  if checkbox1.checked then
    NN.EnableDropouts(true) else
    NN.EnableDropouts(false);
  pInput := TNNetVolume.Create(32, 32, 3, 1);
  pOutPut := TNNetVolume.Create(10, 1, 1, 1);
  LoadPictureIntoVolume(Image1.Picture, pInput);
  if checkbox2.checked then begin
     NN.LoadFromFile(edit1.Text);
     pInput := TNNetVolume.Create(40, 40, 3, 1);
     //pinput.ReSize(40,40,3);
     with image1 do begin
    //AnchorSideLeft.Control:= Owner
      Height:= 40;  //32
      Width:= 40;   //32
     stretch:= true;
     end;
    // LoadImageIntoVolume(TFPMemoryImage(Image1), pInput);
     LoadPictureIntoVolume(Image1.picture, pInput);
     pinput.ReSize(40,40,3);
  end;
  pInput.RgbImgToNeuronalInput(csEncodeRGB);
  NN.Compute(pInput);
  NN.GetOutput(pOutPut);
  Label1.Caption := csTinyImageLabel[pOutPut.GetClass()];
  for k := 1 to 10 do
    StringGrid1.Cells[1, k]:= FloatToStr(NN.Layers[NN.Layers.Count-2].Output.Raw[k - 1]);
    stringgrid1.selection:=
                  TGridRect(Rect(0,pOutPut.GetClass+1,0,pOutPut.GetClass+1));

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
var
  k: integer;
begin
  edit1.text:= '.\model\ClassifyCNNModel_70.nn';
  for k := 0 to 9 do
    StringGrid1.Cells[0, k+1] := csTinyImageLabel[k];
  FindAllFiles(ComboBox1.Items, 'data');
  if ComboBox1.Items.Count > 0 then
  begin
    ComboBox1.Caption := ComboBox1.Items[0];
    if FileExists(ComboBox1.Caption) then
      Image1.Picture.LoadFromFile(ComboBox1.Caption);
  end;
end;

end.
