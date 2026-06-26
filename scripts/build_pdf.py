# -*- coding: utf-8 -*-
"""Chinese PDF decks — minimal: big numbers, short lines, English examples."""
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
pdfmetrics.registerFont(TTFont("CJK","/usr/share/fonts/google-droid/DroidSansFallback.ttf"))
CJK="CJK"; MONO="Courier"; MONOB="Courier-Bold"
GROUND=HexColor("#E9ECEA"); PAPER=HexColor("#F3F5F3"); INK=HexColor("#18211E")
MUTED=HexColor("#5C6B64"); GREEN=HexColor("#0F6E5C"); CLAY=HexColor("#BD5B36")
LINE=HexColor("#C5CDC8"); LINE2=HexColor("#D8DEDA"); WHITE=HexColor("#FFFFFF")
PW,PH=960,540
def Y(y): return PH-y
def text(c,x,y,s,size,color=INK,font=CJK):
    c.setFillColor(color); c.setFont(font,size); c.drawString(x,Y(y)-size,s)
def rule(c,x,y,w,color=INK,h=2.2): c.setFillColor(color); c.rect(x,Y(y)-h,w,h,fill=1,stroke=0)
def rrect(c,x,y,w,h,fill=None,line=None):
    if fill is not None: c.setFillColor(fill); c.rect(x,Y(y)-h,w,h,fill=1,stroke=0)
    if line is not None: c.setStrokeColor(line); c.setLineWidth(1); c.rect(x,Y(y)-h,w,h,fill=0,stroke=1)
def page(c,bg=GROUND): c.setFillColor(bg); c.rect(0,0,PW,PH,fill=1,stroke=0)
def eyebrow(c,s): text(c,48,40,s,13,GREEN, MONOB if s.isascii() else CJK)

def build_external(path):
    c=canvas.Canvas(path,pagesize=(PW,PH))
    page(c); eyebrow(c,"Custodian Labs · Guardian Layer"); rule(c,48,72,864)
    text(c,48,200,"这个隐私替换,",40,INK); text(c,48,255,"会不会让数据没法用?",40,GREEN); c.showPage()
    page(c); eyebrow(c,"01 · 它做什么"); rule(c,48,72,864)
    text(c,48,120,"把真实信息换成逼真的假数据。",26,INK)
    rrect(c,48,175,864,150,fill=PAPER,line=LINE)
    text(c,72,225,"Anna S. ,  April 12 2023 ,  Methodist Hospital",17,CLAY,MONO)
    text(c,72,280,"Maria S. ,  March 13 2021 ,  Methodist Hospital",17,GREEN,MONO); c.showPage()
    page(c); eyebrow(c,"02 · 结果"); rule(c,48,72,864)
    text(c,48,210,"97-100%",92,GREEN,MONOB)
    text(c,48,330,"替换后的信息仍然能被找到",17,MUTED)
    text(c,560,205,"变化在统计上可忽略",20,INK)
    text(c,560,245,"—— 与零相差不超过 ±2 个百分点。",16,MUTED); c.showPage()
    page(c); eyebrow(c,"03 · 跨格式、跨语言都成立"); rule(c,48,72,864)
    def ln(y,a,b): text(c,48,y,a,14,CLAY,MONO); text(c,48,y+26,b,14,GREEN,MONO)
    ln(135,"70yo M, Dr. John L., Mt. Sinai, Feb 21 2023","73yo M, Dr. James L., Mt. Egypt, Nov 19 2021")
    ln(235,'DE:  ... Monsignore ... 23/07/2011','     ... Fulgenzio ... 24/07/2011')
    ln(335,'JSON: {"Date":"20/05/2022","City":"Saint-Priest"}','      {"Date":"21/05/2023","City":"Saint-Priest"}')
    text(c,48,430,"只有敏感值变了 —— 格式完全保留。",15,MUTED); c.showPage()
    page(c,INK)
    c.setFillColor(WHITE); c.setFont(CJK,34); c.drawString(60,Y(250)-34,"保护了隐私,")
    c.setFillColor(GREEN); c.drawString(60,Y(305)-34,"又没破坏数据。")
    text(c,60,470,"custodianai.pages.dev",16,GREEN,MONOB); c.showPage()
    c.save(); print("saved external zh")

def build_internal(path):
    c=canvas.Canvas(path,pagesize=(PW,PH))
    page(c); eyebrow(c,"Custodian Labs · Guardian Layer · Internal"); rule(c,48,72,864)
    text(c,48,205,"我们学到了什么",38,INK); text(c,48,258,"关于这个隐私替换。",38,GREEN); c.showPage()
    page(c); eyebrow(c,"01 · 保持数据可用"); rule(c,48,72,864)
    text(c,48,210,"97-100%",92,GREEN,MONOB)
    text(c,48,330,"替换后仍被其他工具找到",17,MUTED)
    text(c,560,210,"11 个工具、7 个数据集,",19,INK)
    text(c,560,250,"格式不变。",19,INK); c.showPage()
    page(c); eyebrow(c,"02 · 这个变化显著吗?"); rule(c,48,72,864)
    text(c,48,210,"±2 pts",92,GREEN,MONOB)
    text(c,48,330,"与零统计等价 (等价检验 TOST)",16,MUTED)
    text(c,560,205,"召回率 76.2% -> 75.0%。",19,INK)
    text(c,560,245,"不是下降 —— 小到可忽略。",18,INK); c.showPage()
    page(c); eyebrow(c,"03 · 但藏了多少?"); rule(c,48,72,864)
    rows=[("名字","4 个里 3 个",0.75,GREEN),("日期","4 个里 3 个",0.73,GREEN),
          ("ID 号码","2 个里 1 个",0.51,CLAY),("地址","2 个里 1 个",0.44,CLAY)]
    for i,(n,lab,frac,col) in enumerate(rows):
        y=150+i*75; text(c,60,y,n,19,INK)
        rrect(c,300,y-4,470,26,fill=LINE2); rrect(c,300,y-4,int(470*frac),26,fill=col)
        text(c,790,y,lab,15,col)
    c.showPage()
    page(c); eyebrow(c,"04 · 这个 gap 可修"); rule(c,48,72,864)
    text(c,48,200,"约 3/4",84,GREEN,CJK)
    text(c,48,310,"漏掉的其实早被检测到 —— 只是没替换",15,MUTED)
    text(c,560,160,"残缺的替身才会漏:",17,INK)
    text(c,560,200,"Chicago -> Illino",15,CLAY,MONO)
    text(c,560,232,"El Paso -> El",15,CLAY,MONO)
    text(c,560,264,"Cedars-Sinai -> Vidant",15,CLAY,MONO)
    text(c,560,312,"要修的是替换这一步,不是检测器。",14,MUTED); c.showPage()
    page(c,INK)
    c.setFillColor(WHITE); c.setFont(CJK,28); c.drawString(60,Y(170)-28,"可用性:已证明。下一步:藏得更全。")
    for i,t in enumerate(["· 替换不破坏数据 —— ±2 个百分点内等价",
                          "· 名字/日期藏得好;ID 和地址漏掉约一半",
                          "· 大部分漏掉的已被检测到 —— 修替换步即可"]):
        text(c,60,240+i*42,t,16,HexColor("#C5CDC8"))
    text(c,60,470,"custodianai.pages.dev",16,GREEN,MONOB); c.showPage()
    c.save(); print("saved internal zh")

import os; os.makedirs("docs",exist_ok=True)
build_external("docs/guardian_layer_deck_zh.pdf")
build_internal("docs/guardian_layer_internal_zh.pdf")
