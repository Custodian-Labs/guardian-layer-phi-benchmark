"""Internal deck — minimal: big numbers, short phrases, one example per point."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
GROUND=RGBColor(0xE9,0xEC,0xEA); PAPER=RGBColor(0xF3,0xF5,0xF3); INK=RGBColor(0x18,0x21,0x1E)
MUTED=RGBColor(0x5C,0x6B,0x64); GREEN=RGBColor(0x0F,0x6E,0x5C); CLAY=RGBColor(0xBD,0x5B,0x36)
LINE=RGBColor(0xC5,0xCD,0xC8); LINE2=RGBColor(0xD8,0xDE,0xDA); WHITE=RGBColor(0xFF,0xFF,0xFF)
SERIF="Georgia"; MONO="Consolas"
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
s=slide(); eyebrow(s,"Custodian Labs · Guardian Layer · Internal"); rule(s,Inches(1.0))
_,tf=box(s,Inches(0.7),Inches(2.7),Inches(11.9),Inches(2))
para(tf,"What we learned",44,INK,SERIF,b=True,first=True,sa=2)
para(tf,"about the privacy swap.",44,GREEN,SERIF,b=True,it=True)

# 2 GOOD NEWS
s=slide(); eyebrow(s,"01 · Keeps data usable"); rule(s,Inches(1.0))
_,tf=box(s,Inches(0.7),Inches(2.6),Inches(7),Inches(2))
para(tf,"97–100%",96,GREEN,MONO,b=True,first=True,sa=6)
para(tf,"swapped info still found by other tools",18,MUTED,SERIF)
_,tf=box(s,Inches(8.0),Inches(3.1),Inches(4.6),Inches(1.5))
para(tf,"Across 11 tools,",19,INK,SERIF,b=True,first=True,sa=2)
para(tf,"7 datasets. Format intact.",19,INK,SERIF)

# 3 STAT (equivalence)
s=slide(); eyebrow(s,"02 · Is the change significant?"); rule(s,Inches(1.0))
_,tf=box(s,Inches(0.7),Inches(2.6),Inches(7),Inches(2))
para(tf,"±2 pts",96,GREEN,MONO,b=True,first=True,sa=6)
para(tf,"statistically equivalent to zero (TOST)",18,MUTED,SERIF)
_,tf=box(s,Inches(8.0),Inches(3.0),Inches(4.7),Inches(2))
para(tf,"Recall 76.2% → 75.0%.",19,INK,SERIF,b=True,first=True,sa=2)
para(tf,"Not a drop — too small",19,INK,SERIF,sa=2)
para(tf,"to matter.",19,INK,SERIF)

# 4 THE GAP (bars)
s=slide(); eyebrow(s,"03 · But how much does it hide?"); rule(s,Inches(1.0))
rows=[("Names","3 of 4",0.75,GREEN),("Dates","3 of 4",0.73,GREEN),
      ("ID numbers","1 of 2",0.51,CLAY),("Addresses","1 of 2",0.44,CLAY)]
top=Inches(1.9)
for i,(name,lab,frac,col) in enumerate(rows):
    y=top+Inches(1.05)*i
    _,t1=box(s,Inches(0.9),y,Inches(3.4),Inches(0.5)); para(t1,name,20,INK,SERIF,b=True,first=True)
    rect(s,Inches(4.4),y+Inches(0.03),Inches(5.6),Inches(0.42),fill=LINE2)
    rect(s,Inches(4.4),y+Inches(0.03),Inches(5.6*frac),Inches(0.42),fill=col)
    _,t2=box(s,Inches(10.3),y,Inches(2.4),Inches(0.5)); para(t2,lab,18,col,MONO,b=True,first=True)

# 5 FIXABLE
s=slide(); eyebrow(s,"04 · The gap is fixable"); rule(s,Inches(1.0))
_,tf=box(s,Inches(0.7),Inches(2.3),Inches(6),Inches(2))
para(tf,"~3 in 4",90,GREEN,MONO,b=True,first=True,sa=6)
para(tf,"misses were already detected — just not swapped",17,MUTED,SERIF)
_,tf=box(s,Inches(7.4),Inches(2.4),Inches(5.3),Inches(2.6))
para(tf,"Broken surrogates miss:",17,INK,SERIF,b=True,first=True,sa=8)
para(tf,"Chicago → Illino",15,CLAY,MONO,sa=4)
para(tf,"El Paso → El",15,CLAY,MONO,sa=4)
para(tf,"Cedars-Sinai → Vidant",15,CLAY,MONO,sa=8)
para(tf,"Fix the swap step, not the detector.",15,MUTED,SERIF)

# 6 BOTTOM LINE
s=slide(bg=INK)
_,tf=box(s,Inches(0.9),Inches(2.4),Inches(11.5),Inches(3))
para(tf,"Usable: proven. Hide more: next.",32,WHITE,SERIF,b=True,first=True,sa=20)
para(tf,"✓  Swap doesn’t break data — equivalent within ±2 pts",18,RGBColor(0xC5,0xCD,0xC8),SERIF,sa=10)
para(tf,"✓  Hides names/dates well; misses ~half of IDs & addresses",18,RGBColor(0xC5,0xCD,0xC8),SERIF,sa=10)
para(tf,"✓  Most misses already detected — fixable in the swap step",18,RGBColor(0xC5,0xCD,0xC8),SERIF)
_,tf=box(s,Inches(0.9),Inches(6.2),Inches(11.5),Inches(0.5))
para(tf,"custodianai.pages.dev",16,GREEN,MONO,b=True,first=True)

import os; os.makedirs("docs",exist_ok=True)
prs.save("docs/guardian_layer_internal.pptx")
print("saved internal ·",len(prs.slides._sldIdLst),"slides")
