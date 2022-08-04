unit test;

interface

uses
  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants, System.Classes, Vcl.Graphics,
  Vcl.Controls, Vcl.Forms, Vcl.Dialogs, Vcl.StdCtrls, Vcl.ExtCtrls,
  PythonEngine, WrapDelphi;

type
  TForm1 = class(TForm)
    Image1: TImage;
    Timer1: TTimer;
    btnStart: TButton;
    PythonEngine1: TPythonEngine;
    PythonModule1: TPythonModule;
    PythonType1: TPythonType;
    PythonDelphiVar1: TPythonDelphiVar;
    PyDelphiWrapper1: TPyDelphiWrapper;
    PythonInputOutput1: TPythonInputOutput;
  private
    { Private declarations }
  public
    { Public declarations }
  end;

var
  Form1: TForm1;

implementation

{$R *.dfm}

end.
