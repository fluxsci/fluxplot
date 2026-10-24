"""Author the additive layout contract in manifest.schema.json (idempotent)."""
import json
from pathlib import Path

N = {"type": "number"}
S = {"type": "string"}
B = {"type": "boolean"}
NULL = {"type": "null"}
def arr(item=N): return {"type": "array", "items": item}
def obj(props, required=()):
    return {"type": "object", "properties": props, **({"required": list(required)} if required else {})}
def dictionary(item): return {"type": "object", "additionalProperties": item}
def nullable(item): return {"anyOf": [item, NULL]}
def enum(*items): return {"enum": list(items)}
def ref(name): return {"$ref": "#/definitions/"+name}
def named(name, value): return {"title": name, **value}

slot = named("LayoutSlot", obj(dict(grid=S, rows=arr(), cols=arr()), ("grid", "rows", "cols")))
locator = named("LayoutLocator", obj(dict(kind=enum("maxn", "multiple", "log", "null", "auto-minor", "fixed", "category", "date", "custom"),
    nbins={"anyOf": [N, enum("auto")]}, steps=arr(), integer=B, symmetric=B, prune=nullable(S), minNTicks=N,
    base=N, offset=N, subs={"anyOf": [arr(), S, NULL]}, numticks={"anyOf": [N, S]}, n={"anyOf":[N,enum("auto"),NULL]}), ("kind",)))
formatter = named("LayoutFormatter", obj(dict(kind=enum("scalar", "log-sci", "percent", "str-method", "fixed", "custom"),
    useOffset=B, offset=nullable(N), offsetThreshold=N, useMathText=B, powerlimits=arr(), scientific=B,
    useLocale=B, unicodeMinus=B, base=N, labelOnlyBase=B, minorThresholds=arr({"anyOf":[N,enum("inf","-inf")]}),minExponent=N, xmax=N, decimals=nullable(N), symbol=S, fmt=S), ("kind",)))
style = named("LayoutAxisStyle", obj(dict(side=S, labelSide=S, tickDir=S, tickLenPt=N, tickPadPt=N,
    minorTickLenPt=N, labelPadPt=N, labelAuto=B, font=S, labelRotation=N, labelHa=S,
    offsetText=nullable(S),offsetSpec=obj(dict(id=nullable(S),anchor=arr(),font=S,ha=S,va=S,rotation=N),("id","anchor","font","ha","va","rotation")), label=nullable(S), visible=B, tickTop=B, labelTop=B, tickBottom=B, labelBottom=B),
    ("side", "font", "tickLenPt", "tickPadPt", "labelPadPt", "visible")))
ticking = named("LayoutTicking", obj(dict(major=ref("layoutLocator"), minor=ref("layoutLocator"), formatter=ref("layoutFormatter"),
    retick=B,minorFormatter=ref("layoutFormatter"),minorValues=arr(),minorLabels=arr(S),minorRetick=B, domain=arr(), scale=S, values=arr(), labels=arr(S), offsetText=S), ("major", "minor", "formatter", "retick", "domain", "scale", "values")))
axis_props = dict(id=S, slot=nullable(ref("layoutSlot")), position=arr(), initialPosition=arr(), extraLayoutIds=arr(S), positionMode=enum("grid", "manual", "inset"),
    initialDomain=dictionary(arr()),dataLim=dictionary(arr(nullable(N))),
    inset=nullable(obj(dict(parent=nullable(S), bounds=arr(), frame=S))), inLayout=B, aspect={"anyOf": [N,S]},dataRatio=N,dataAspect={"anyOf":[N,S]},managed=B,
    adjustable=S, anchor={"anyOf": [S,arr()]}, boxAspect=nullable(N), twins=arr(obj(dict(id=S,axis=S))),
    sharex=nullable(S), sharey=nullable(S), title=obj(dict(left=nullable(S),center=nullable(S),right=nullable(S),padPt=N)),
    axisStyle=dictionary(ref("layoutAxisStyle")), ticking=dictionary(ref("layoutTicking")), margins=arr(), pixelBox=arr())
axis = named("LayoutAxis", obj(axis_props, ("id", "slot", "position", "positionMode", "inLayout", "aspect", "adjustable", "anchor", "title", "axisStyle", "ticking", "margins", "pixelBox")))
cb = named("LayoutColorbar", obj({**axis_props, **dict(parents=arr(S), location=S, orientation=S, fraction=N, shrink=N,
    padFrac=N, panchor=arr(), extendFrac={"anyOf": [N,arr(),S,NULL]})}, ("id", "parents", "location", "orientation", "fraction", "shrink", "aspect", "padFrac", "axisStyle", "ticking", "pixelBox", "margins")))
endpoint = named("LayoutEndpoint", obj(dict(f=enum("data", "axes", "figure"), axes=nullable(S), pt=arr(), nativePt=nullable(arr())), ("f", "pt")))
frame = named("LayoutFrame", obj(dict(f=enum("data", "glyph", "blend", "image", "gradient", "clip", "axes", "offset", "arrow", "figure", "axis", "title", "legend"),
    box=obj(dict(x=N,y0=N,y1=N,notch=nullable(N),widthPt=N,depthPt=N,heightPt=N)),
    mark=obj(dict(orientation=enum("vertical", "horizontal"), position=N, mean=N, bound=nullable(N),
        radiusPt=N, capPt=nullable(N)), ("orientation", "position", "mean", "bound", "radiusPt", "capPt")),
    anchorsPt=arr(arr()), axes=nullable(S), anchor=S, frac=arr(), geometry=B, x=S, y=S, base=ref("layoutEndpoint"), offsetPt=arr(),
    textcoords=S, text=S, xy=ref("layoutEndpoint"), shrinkA=N, shrinkB=N, style=S, connection=S,
    axis=S, part=S, value=nullable(N), minor=B, side=S, position=arr({"anyOf": [S,N]}), slot=S), ("f",)))
layout = named("FluxPlotLayout", obj(dict(version={"const": 1}, supported=B, reasons=arr(S),
    figure=obj(dict(widthPt=N,heightPt=N,suptitle=nullable(S),supxlabel=nullable(S),supylabel=nullable(S),auto=dictionary(B)), ("widthPt", "heightPt")),
    engine=obj(dict(kind=enum("none","tight","constrained"), wPad=nullable(N),hPad=nullable(N),wSpace=N,hSpace=N,
        rect=arr(),pad=N,fontPt=N,subplotpars=obj({k:N for k in ("left","right","bottom","top","wspace","hspace")})), ("kind",)),
    grids=arr(obj(dict(id=S,nrows=N,ncols=N,widthRatios=arr(),heightRatios=arr(),wspace=nullable(N),hspace=nullable(N),parent=nullable(ref("layoutSlot"))), ("id","nrows","ncols","widthRatios","heightRatios"))),
    axes=arr(ref("layoutAxis")),colorbars=arr(ref("layoutColorbar")),
    legends=arr(obj(dict(collision=nullable(obj(dict(boxes=arr(arr()),lines=arr(arr(arr())),points=arr(arr())))),id=S,owner=S,locRequested=S,locResolved=S,bboxToAnchor=nullable(obj(dict(bbox=arr(),frame=S))),borderAxesPadPt=N,inLayout=B,sizePt=arr(),bbox=arr()), ("id","owner","locRequested","locResolved","sizePt","bbox","borderAxesPadPt"))),
    texts=dictionary(obj(dict(bbox=arr(),anchor=arr(),ha=S,va=S,rotation=N,font=S,text=S), ("bbox","anchor","ha","va","rotation","font","text"))),
    frames=dictionary(ref("layoutFrame")),fonts=dictionary(obj(dict(family=S,sizePt=N,weight={"anyOf":[N,S]},style=S,ascentPt=N,descentPt=N,advances=dictionary(N),math=obj(dict(powerRawWidthPt=N,supAdvancePt=N,minusAdvancePt=N,quantumPt=N,supScale=N,supRisePt=N,baseAdvance10=N,powerWidthPt=N,powerHeightPt=N,powerDescentPt=N,coeffPrefixPt=N,timesPadPt=N))), ("family","sizePt","ascentPt","descentPt","advances")))),
    ("version","supported","reasons","figure","engine","grids","axes","colorbars","legends","texts","frames","fonts")))

if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1]/"src/fluxplot/schemas/manifest.schema.json"
    doc = json.loads(path.read_text())
    doc["$id"] = "https://flux.dev/schemas/fluxplot-manifest-0.4.0.json"
    doc["properties"]["layout"] = ref("layout")
    doc.setdefault("definitions", {}).update(dict(layout=layout, layoutSlot=slot, layoutAxis=axis,
        layoutColorbar=cb, layoutLocator=locator, layoutFormatter=formatter, layoutAxisStyle=style,
        layoutTicking=ticking, layoutFrame=frame, layoutEndpoint=endpoint))
    path.write_text(json.dumps(doc, indent=2)+"\n")
    recipe = path.with_name("recipe.schema.json")
    doc = json.loads(recipe.read_text())
    doc["$id"] = "https://flux.dev/schemas/fluxplot-recipe-0.4.0.json"
    recipe.write_text(json.dumps(doc, indent=2)+"\n")
