"""External deck — minimal: big numbers, one example per slide, no paragraphs."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
GROUND=RGBColor(0xE9,0xEC,0xEA); PAPER=RGBColor(0xF3,0xF5,0xF3); INK=RGBColor(0x18,0x21,0x1E)
MUTED=RGBColor(0x5C,0x6B,0x64); GREEN=RGBColor(0x0F,0x6E,0x5C); CLAY=RGBColor(0xBD,0x5B,0x36)
LINE=RGBColor(0xC5,0xCD,0xC8); WHITE=RGBColor(0xFF,0xFF,0xFF); SERIF="Georgia"; MONO="Consolas"
W,H=Inches(13.333),Inches(7.5)
prs=Presentation(); prs.slide_width=W; prs.slide_height=H; BLANK=prs.slide_layouts[6]
def slide(bg=GROUND):
    s=prs.slides.add_slide(BLANK); r=s.shapes.add_shape(1,0,0,W,H)
    r.fill.solid(); r.fill.fore_color.rgb=bg; r.line.fill.background(); r.shadow.inherit=False
    s.shapes._spTree.remove(r._element); s.shapes._spTree.insert(2,r._element); return s
def box(s,x,y,w,h):
    tb=s.shapes.add_textbox(x,y,w,h); tf=tb.text_frame; tf.word_wrap=True
    tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0; return tb,tf
def para(tf,t,sz,c=INK,fn=SERIF,b=False,it=False,sa=6,first=False):
    p=tf.paragraphs[0] if first else tf.add_paragraph(); p.space_after=Pt(sa)
    r=p.add_run(); r.text=t; f=r.font; f.size=Pt(sz); f.name=fn; f.bold=b; f.italic=it; f.color.rgb=c
def rect(s,x,y,w,h,fill=None,line=None):
    sp=s.shapes.add_shape(1,x,y,w,h)
    if fill is None: sp.fill.background()
    else: sp.fill.solid(); sp.fill.fore_color.rgb=fill
    if line is None: sp.line.fill.background()
    else: sp.line.color.rgb=line; sp.line.width=Pt(1)
    sp.shadow.inherit=False
def eyebrow(s,t): _,tf=box(s,Inches(0.7),Inches(0.55),Inches(11.9),Inches(0.4)); para(tf,t.upper(),12,GREEN,MONO,b=True,first=True)
def rule(s,y): rect(s,Inches(0.7),y,Inches(11.9),Pt(2.2),fill=INK)

# 1 TITLE
s=slide(); eyebrow(s,"Custodian Labs · Guardian Layer"); rule(s,Inches(1.0))
_,tf=box(s,Inches(0.7),Inches(2.6),Inches(11.9),Inches(2.2))
para(tf,"Does the privacy swap",44,INK,SERIF,b=True,first=True,sa=2)
para(tf,"keep data usable?",44,GREEN,SERIF,b=True,it=True)

# 2 WHAT IT DOES — one example
s=slide(); eyebrow(s,"01 · What it does"); rule(s,Inches(1.0))
_,tf=box(s,Inches(0.7),Inches(1.5),Inches(11.9),Inches(0.6))
para(tf,"It swaps real info for a realistic fake.",26,INK,SERIF,b=True,first=True)
rect(s,Inches(0.7),Inches(2.5),Inches(11.9),Inches(1.9),fill=PAPER,line=LINE)
_,tf=box(s,Inches(1.0),Inches(2.95),Inches(11.3),Inches(1.4))
para(tf,"Anna S. ,  April 12 2023 ,  Methodist Hospital",18,CLAY,MONO,first=True,sa=18)
para(tf,"Maria S. ,  March 13 2021 ,  Methodist Hospital",18,GREEN,MONO)

# 3 RESULT — big number
s=slide(); eyebrow(s,"02 · The result"); rule(s,Inches(1.0))
_,tf=box(s,Inches(0.7),Inches(2.3),Inches(7),Inches(2.4))
para(tf,"97–100%",96,GREEN,MONO,b=True,first=True,sa=6)
para(tf,"the swapped info is still found",18,MUTED,SERIF)
_,tf=box(s,Inches(8.0),Inches(2.7),Inches(4.6),Inches(2))
para(tf,"Change is statistically",19,INK,SERIF,b=True,first=True,sa=2)
para(tf,"negligible — within ±2 pts.",19,INK,SERIF)

# 4 EVERYWHERE — 3 one-line examples
s=slide(); eyebrow(s,"03 · Holds across formats & languages"); rule(s,Inches(1.0))
def line(y,a,b):
    _,tf=box(s,Inches(0.7),y,Inches(11.9),Inches(0.6))
    para(tf,a,15,CLAY,MONO,first=True,sa=2); para(tf,b,15,GREEN,MONO)
line(Inches(1.7),"70yo M, Dr. John L., Mt. Sinai, Feb 21 2023","73yo M, Dr. James L., Mt. Egypt, Nov 19 2021")
line(Inches(3.2),'German:  ... Monsignore ... 23/07/2011','        ... Fulgenzio ... 24/07/2011')
line(Inches(4.7),'JSON:  {"Date":"20/05/2022","City":"Saint-Priest"}','       {"Date":"21/05/2023","City":"Saint-Priest"}')
_,tf=box(s,Inches(0.7),Inches(6.0),Inches(11.9),Inches(0.5))
para(tf,"Only the private value changes — the format stays exact.",15,MUTED,SERIF,first=True)

# 5 BOTTOM LINE
s=slide(bg=INK)
_,tf=box(s,Inches(0.9),Inches(3.0),Inches(11.5),Inches(1.6))
para(tf,"Protects privacy.",36,WHITE,SERIF,b=True,first=True,sa=4)
para(tf,"Doesn’t break the data.",36,GREEN,SERIF,b=True,it=True)
_,tf=box(s,Inches(0.9),Inches(6.0),Inches(11.5),Inches(0.5))
para(tf,"custodianai.pages.dev",16,GREEN,MONO,b=True,first=True)

import os; os.makedirs("docs",exist_ok=True)
prs.save("docs/guardian_layer_deck.pptx")
print("saved external ·",len(prs.slides._sldIdLst),"slides")
