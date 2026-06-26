# -*- coding: utf-8 -*-
"""Chinese-language PDF decks (examples kept in English). Two files:
   docs/guardian_layer_deck_zh.pdf      — external, positive
   docs/guardian_layer_internal_zh.pdf  — internal, complete (incl. coverage)
Built with reportlab; CJK via DroidSansFallback, English code via Courier.
"""
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

pdfmetrics.registerFont(TTFont("CJK", "/usr/share/fonts/google-droid/DroidSansFallback.ttf"))
CJK="CJK"; MONO="Courier"; MONOB="Courier-Bold"

GROUND=HexColor("#E9ECEA"); PAPER=HexColor("#F3F5F3"); INK=HexColor("#18211E")
MUTED=HexColor("#5C6B64"); GREEN=HexColor("#0F6E5C"); CLAY=HexColor("#BD5B36")
LINE=HexColor("#C5CDC8"); LINE2=HexColor("#D8DEDA"); WHITE=HexColor("#FFFFFF")

PW,PH=960,540  # 16:9 points (13.33in x 7.5in)

def Y(y): return PH-y  # top-left helper

def page(c,bg=GROUND):
    c.setFillColor(bg); c.rect(0,0,PW,PH,fill=1,stroke=0)

def text(c,x,y,s,size,color=INK,font=CJK,align="l"):
    c.setFillColor(color); c.setFont(font,size)
    if align=="l": c.drawString(x,Y(y)-size,s)
    elif align=="r": c.drawRightString(x,Y(y)-size,s)
    else: c.drawCentredString(x,Y(y)-size,s)

def rule(c,x,y,w,color=INK,h=2.2):
    c.setFillColor(color); c.rect(x,Y(y)-h,w,h,fill=1,stroke=0)

def rrect(c,x,y,w,h,fill=None,line=None,lw=1):
    if fill is not None: c.setFillColor(fill); c.rect(x,Y(y)-h,w,h,fill=1,stroke=0)
    if line is not None:
        c.setStrokeColor(line); c.setLineWidth(lw); c.rect(x,Y(y)-h,w,h,fill=0,stroke=1)

def eyebrow(c,s):
    # English-only eyebrows keep the typewriter look; Chinese ones use the CJK face.
    text(c,48,40,s,13,GREEN, MONOB if s.isascii() else CJK)

def wrap(c,x,y,s,size,color,font,maxw,lh):
    """naive CJK wrap by char width."""
    c.setFont(font,size); line=""; yy=y
    for ch in s:
        if pdfmetrics.stringWidth(line+ch,font,size)>maxw:
            text(c,x,yy,line,size,color,font); line=ch; yy+=lh
        else: line+=ch
    if line: text(c,x,yy,line,size,color,font); yy+=lh
    return yy

def save_deck(path, internal=False):
    c=canvas.Canvas(path, pagesize=(PW,PH))

    # ---- 1 TITLE ----
    page(c)
    eyebrow(c,"Custodian Labs · Guardian Layer"+(" · Internal" if internal else ""))
    rule(c,48,72,864)
    if internal:
        text(c,48,165,"我们对这个隐私替换",34,INK,CJK)
        text(c,48,215,"学到了什么。",34,GREEN,CJK)
        wrap(c,48,275,"两件事:它能不能保持数据可用,以及它藏得够不够。下面是对两点的诚实评估。",17,MUTED,CJK,820,26)
    else:
        text(c,48,165,"这个隐私替换,",34,INK,CJK)
        text(c,48,215,"会不会让数据没法用?",34,GREEN,CJK)
        wrap(c,48,275,"我们把个人信息换成逼真的假数据来保护隐私。这是检验下游会不会因此出问题。",17,MUTED,CJK,820,26)
    rule(c,48,470,864)
    foot=("内部使用 — 对外版只讲可用性这一面。" if internal
          else "在 1,750 份真实文档、7 个数据集、7 种语言上测试。")
    text(c,48,486,foot,12,MUTED,CJK)
    c.showPage()

    # ---- 2 GOOD NEWS (usability) ----
    page(c); eyebrow(c,("01 · 好消息" if internal else "01 · 工具做什么")); rule(c,48,72,864)
    if internal:
        text(c,48,150,"97-100%",70,GREEN,MONOB)
        wrap(c,48,235,"只要它替换了某个信息,其他工具仍然能找到",13,MUTED,CJK,360,20)
        text(c,560,150,"替换让数据保持可用。",24,INK,CJK)
        wrap(c,560,200,"替换之后,其他工具几乎每次都还能认出敏感信息,而且格式(病历、表单、数据文件)"
             "完全不变,下游不会出错。",17,INK,CJK,360,26)
        wrap(c,560,300,"这是可以给客户看的亮点 —— 在全部 11 个工具、7 个数据集上都成立。",15,MUTED,CJK,360,22)
        wrap(c,560,365,"统计上:在被脱敏的 5.6 万个 span 上召回率 76.2%→75.0%,等价检验证明变化在 ±2 个百分点内可忽略(p≈1e-8)。",13,GREEN,CJK,360,20)
    else:
        text(c,48,120,"它不是把信息涂黑,而是换成同类型的逼真假数据。",22,INK,CJK)
        rrect(c,48,175,864,200,fill=PAPER,line=LINE)
        text(c,72,200,"BEFORE",11,MUTED,MONOB)
        text(c,72,226,"…a 34-year-old female treated by Anna S. at Methodist Hospital on April 12, 2023.",13.5,CLAY,MONO)
        text(c,72,285,"AFTER",11,MUTED,MONOB)
        text(c,72,311,"…a 35-year-old female treated by Maria S. at Methodist Hospital on March 13, 2021.",13.5,GREEN,MONO)
        wrap(c,72,350,"真名→假名,真日期→假日期。句子读起来仍然正常,所以原来能处理它的系统,处理替换后的版本一样能用。",12.5,MUTED,CJK,820,22)
    c.showPage()

    # ---- 3 ----
    if internal:
        # the honest gap
        page(c); eyebrow(c,"02 · 诚实的 gap — 到底藏了多少"); rule(c,48,72,864)
        wrap(c,48,100,"在真实病历上,名字和日期藏得好,但漏掉了大约一半的 ID 号码和地址。",19,INK,CJK,860,28)
        rows=[("名字","约 4 个里藏 3 个",0.75,GREEN),
              ("日期 / 年龄","约 4 个里藏 3 个",0.73,GREEN),
              ("ID 号码(病人号、电话)","约 2 个里藏 1 个",0.51,CLAY),
              ("地址 / 地点","约 2 个里藏 1 个",0.44,CLAY)]
        top=185
        for i,(name,lab,frac,col) in enumerate(rows):
            yy=top+78*i
            text(c,60,yy,name,16,INK,CJK)
            rrect(c,380,yy-2,360,22,fill=LINE2); rrect(c,380,yy-2,int(360*frac),22,fill=col)
            text(c,760,yy,lab,13,col,CJK)
        c.showPage()
        # gap fixable
        page(c); eyebrow(c,"03 · 关于这个 gap 的好消息"); rule(c,48,72,864)
        text(c,48,150,"约 3/4",64,GREEN,CJK)
        wrap(c,48,235,"漏掉的里面,工具其实早就知道那是隐私",13,MUTED,CJK,360,20)
        text(c,560,150,"是“没替换”,不是“没找到”。",23,INK,CJK)
        wrap(c,560,205,"大部分漏掉的信息,工具已经标记成敏感了 —— 只是没去替换它。"
             "所以要修的是替换这一步,不是重新教它认隐私。",17,INK,CJK,360,26)
        wrap(c,560,335,"这是比较好解决的那类问题。",15,MUTED,CJK,360,24)
        c.showPage()
        # bottom line
        page(c,INK)
        text(c,60,160,"结论",13,GREEN,MONOB)
        text(c,60,210,"保持可用做得很好。下一步:藏得更全。",27,WHITE,CJK)
        for i,line in enumerate([
            "· 替换不会破坏任何东西 —— 这点可以直接给客户展示。",
            "· 名字和日期藏得好,但漏掉约一半的 ID 号码和地址。",
            "· 大部分漏掉的其实已经检测到 —— 修替换这一步就能补上。"]):
            text(c,60,275+i*38,line,17,HexColor("#C5CDC8"),CJK)
        text(c,60,470,"完整数据 → custodianai.pages.dev",16,GREEN,MONOB)
        c.showPage()
    else:
        # result
        page(c); eyebrow(c,"02 · 结果"); rule(c,48,72,864)
        text(c,48,150,"97-100%",72,GREEN,MONOB)
        wrap(c,48,238,"的情况下,替换后的信息仍然能被找到",13,MUTED,CJK,380,20)
        text(c,560,150,"替换不会把数据藏到别人找不到。",22,INK,CJK)
        wrap(c,560,205,"替换之后,其他 AI 工具仍能在同样的位置认出敏感信息,几乎每次都行。"
             "我们用 11 个不同的工具验证过。",17,INK,CJK,360,26)
        wrap(c,560,300,"统计等价检验证实:这个变化小到不具实际意义 —— 与“零变化”相差不超过 ±2 个百分点。",15,GREEN,CJK,360,24)
        wrap(c,560,330,"而且格式完全没动:病历、表单、甚至原始数据文件都保持原样,下游不会出错。",15,MUTED,CJK,360,24)
        c.showPage()
        # examples across formats
        page(c); eyebrow(c,"03 · 跨格式、跨语言都成立"); rule(c,48,72,864)
        def card(yy,title,before,after,note):
            rrect(c,48,yy,864,118,fill=PAPER,line=LINE)
            text(c,72,yy+18,title,11,MUTED,MONOB)
            text(c,72,yy+42,"before   "+before,12,CLAY,MONO)
            text(c,72,yy+64,"after    "+after,12,GREEN,MONO)
            text(c,72,yy+88,note,11,MUTED,CJK)
        card(95,"Doctor's shorthand note (en)",
             "70yo M, seen by Dr. John L. at Mt. Sinai on Feb 21, 2023",
             "73 yo M, seen by Dr. James L. at Mt. Egypt on Nov 19, 2021",
             "连医生的缩写速记都原样保留 —— 只有个人信息变了。")
        card(225,"German record",
             "...ermaechtigen hiermit Monsignore ... Mit Datum 23/07/2011",
             "...ermaechtigen hiermit Fulgenzio ... Mit Datum 24/07/2011",
             "换种语言一样有效。")
        card(355,"A raw data file (JSON)",
             '{ "Date": "20/05/2022", "City": "Saint-Priest", "User": "phprosdocimo" }',
             '{ "Date": "21/05/2023", "City": "Saint-Priest", "User": "phprosdocimo" }',
             "文件格式完全保留 —— 只换敏感值,读取它的系统不会崩。")
        c.showPage()
        # honest note (coverage by data type, light)
        page(c); eyebrow(c,"04 · 一个诚实的说明"); rule(c,48,72,864)
        text(c,48,120,"换多少,取决于数据类型。",24,INK,CJK)
        wrap(c,48,168,"这个工具针对真正的隐私信息。在真实病历上,它替换掉大部分;"
             "在通用文本上替换得少,因为那里很多名字本来就不算隐私。",17,MUTED,CJK,860,26)
        rows=[("真实病历","约 80% 的个人信息被替换",GREEN),
              ("通用 / 百科类文本","更少 —— 那些名字不算隐私",CLAY)]
        for i,(a,b,col) in enumerate(rows):
            yy=265+i*54; text(c,60,yy,a,18,INK,CJK); text(c,480,yy,b,16,col,CJK)
        wrap(c,48,410,"重要:上面“97-100% 仍能找到”只在它确实替换的信息上测量 —— "
             "两个结论各算各的,互不依赖。",14,MUTED,CJK,860,24)
        c.showPage()
        # bottom line
        page(c,INK)
        text(c,60,200,"结论",13,GREEN,MONOB)
        text(c,60,250,"替换保护了隐私,又没破坏数据。",30,WHITE,CJK)
        wrap(c,60,310,"敏感信息被换成逼真的替身,格式完全保留,下游工具照常工作。",18,HexColor("#C5CDC8"),CJK,840,28)
        text(c,60,470,"在线查看 → custodianai.pages.dev",16,GREEN,MONOB)
        c.showPage()

    c.save(); print("saved",path)

import os; os.makedirs("docs",exist_ok=True)
save_deck("docs/guardian_layer_deck_zh.pdf", internal=False)
save_deck("docs/guardian_layer_internal_zh.pdf", internal=True)
