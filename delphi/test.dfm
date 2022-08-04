object Form1: TForm1
  Left = 0
  Top = 0
  Caption = 'Form1'
  ClientHeight = 589
  ClientWidth = 652
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -11
  Font.Name = 'Tahoma'
  Font.Style = []
  OldCreateOrder = False
  DesignSize = (
    652
    589)
  PixelsPerInch = 96
  TextHeight = 13
  object Image1: TImage
    Left = 8
    Top = 8
    Width = 636
    Height = 369
    Anchors = [akLeft, akTop, akRight]
    ExplicitWidth = 925
  end
  object btnStart: TButton
    Left = 8
    Top = 383
    Width = 175
    Height = 41
    Hint = 'Hint for this button!'
    Caption = 'Start'
    ParentShowHint = False
    ShowHint = True
    Style = bsCommandLink
    TabOrder = 0
  end
  object Timer1: TTimer
    Left = 600
    Top = 24
  end
  object PythonEngine1: TPythonEngine
    Left = 440
    Top = 392
  end
  object PythonModule1: TPythonModule
    Engine = PythonEngine1
    Errors = <>
    Left = 448
    Top = 512
  end
  object PythonType1: TPythonType
    Engine = PythonEngine1
    Prefix = 'Create'
    Services.Basic = [bsGetAttr, bsSetAttr, bsRepr, bsStr]
    Services.InplaceNumber = []
    Services.Number = []
    Services.Sequence = []
    Services.Mapping = []
    Left = 576
    Top = 448
  end
  object PythonDelphiVar1: TPythonDelphiVar
    Engine = PythonEngine1
    Module = '__main__'
    VarName = 'varname1'
    Left = 456
    Top = 448
  end
  object PyDelphiWrapper1: TPyDelphiWrapper
    Engine = PythonEngine1
    Left = 584
    Top = 392
  end
  object PythonInputOutput1: TPythonInputOutput
    UnicodeIO = False
    RawOutput = False
    Left = 576
    Top = 512
  end
end
