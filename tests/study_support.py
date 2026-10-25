"""A small synthetic biochemistry lecture (two topics: glycolysis, collagen) used by the study tests. No audio, no model."""

from __future__ import annotations

from src.study.index import DocumentIndex, build_index

GLYCOLYSIS = [
    "Glycolysis is the metabolic pathway that converts glucose into pyruvate in the cytoplasm of the cell.",
    "The first step of glycolysis is catalyzed by hexokinase, which phosphorylates glucose to glucose-6-phosphate.",
    "Hexokinase uses ATP as the phosphate donor, and this reaction traps glucose inside the cell.",
    "Remember that hexokinase is inhibited by its product glucose-6-phosphate.",
    "Phosphofructokinase is the key regulatory enzyme of glycolysis and it is activated by AMP and fructose-2,6-bisphosphate.",
    "High levels of ATP inhibit phosphofructokinase, which slows glycolysis when the cell has enough energy.",
    "Pyruvate kinase catalyzes the last step of glycolysis and produces pyruvate and ATP.",
    "Pyruvate kinase deficiency causes hemolytic anemia because red blood cells depend on glycolysis for ATP.",
    "This is important for the exam: red blood cells have no mitochondria and therefore rely only on glycolysis.",
    "Net glycolysis gives two ATP and two NADH per glucose molecule.",
    "Under anaerobic conditions lactate dehydrogenase converts pyruvate to lactate and regenerates NAD+.",
    "Lactate dehydrogenase is the enzyme that allows glycolysis to continue without oxygen.",
    "In muscle during intense exercise lactate accumulates and leads to a drop in pH.",
    "GLUT4 is the insulin-dependent glucose transporter of muscle and adipose tissue.",
    "Insulin stimulates the translocation of GLUT4 to the plasma membrane, which increases glucose uptake.",
    "In the liver glucokinase has a high Km and acts as a glucose sensor after meals.",
    "Glucokinase is not inhibited by glucose-6-phosphate, unlike hexokinase.",
    "Patients with diabetes have a reduced uptake of glucose into muscle because GLUT4 translocation is impaired.",
    "The enzyme aldolase splits fructose-1,6-bisphosphate into two triose phosphates.",
    "Triose phosphate isomerase interconverts dihydroxyacetone phosphate and glyceraldehyde-3-phosphate.",
    "Glyceraldehyde-3-phosphate dehydrogenase produces the first NADH of glycolysis.",
    "Arsenic poisoning interferes with glycolysis because arsenate replaces phosphate in this step.",
    "Phosphoglycerate kinase generates the first ATP of glycolysis by substrate level phosphorylation.",
    "Enolase converts 2-phosphoglycerate to phosphoenolpyruvate, a high energy compound.",
    "Fluoride inhibits enolase and is added to blood tubes to stop glycolysis in the sample.",
    "Hexokinase has a low Km for glucose, so it works even when the blood glucose is low.",
    "Substrate level phosphorylation means that ATP is formed by direct transfer of a phosphate group from a substrate.",
    "Mutations of glucokinase cause a form of maturity onset diabetes of the young.",
    "The Warburg effect describes the high rate of glycolysis in cancer cells even in the presence of oxygen.",
    "Do not forget that glycolysis happens in the cytoplasm and not in the mitochondria.",
]

COLLAGEN = [
    "Collagen is the most abundant structural protein of the extracellular matrix.",
    "Each collagen molecule is composed of three polypeptide chains wound into a triple helix.",
    "Glycine is found at every third position of the collagen chain because only glycine fits inside the triple helix.",
    "Proline hydroxylase converts proline residues to hydroxyproline after the chain is made.",
    "Hydroxylation of proline needs vitamin C as a cofactor and molecular oxygen as a substrate.",
    "Vitamin C deficiency causes scurvy because collagen without hydroxyproline is unstable.",
    "Patients with scurvy have bleeding gums, poor wound healing and fragile blood vessels.",
    "Lysyl hydroxylase hydroxylates lysine residues, which are later used to form cross links.",
    "Lysyl oxidase forms the covalent cross links between collagen molecules and needs copper.",
    "Copper deficiency therefore reduces the strength of collagen.",
    "Remember that procollagen is made in the endoplasmic reticulum and secreted from the cell.",
    "Procollagen peptidases remove the propeptides outside the cell, and the collagen molecules assemble into fibrils.",
    "Osteogenesis imperfecta is a genetic disease with mutations in the collagen type one genes.",
    "Patients with osteogenesis imperfecta have brittle bones and blue sclerae.",
    "Ehlers Danlos syndrome is caused by defects in collagen synthesis and leads to hyperextensible joints.",
    "Collagen type one is the main collagen of bone, skin and tendon.",
    "Collagen type two is found in cartilage, while collagen type three is found in blood vessels and skin.",
    "Elastin differs from collagen because it can stretch and recoil.",
    "Fibrillin forms microfibrils that support elastin, and mutations of fibrillin cause Marfan syndrome.",
    "Marfan syndrome leads to long limbs, lens dislocation and aortic aneurysm.",
    "Matrix metalloproteinases degrade collagen and require zinc.",
    "Smoking and ageing increase the breakdown of collagen in the skin.",
    "The triple helix is stabilised by hydrogen bonds between hydroxyproline residues.",
    "Hydroxyproline is therefore a marker of collagen breakdown and can be measured in urine.",
    "Please note that the hydroxylation reaction happens in the lumen of the endoplasmic reticulum.",
    "Glycosylation of hydroxylysine residues is also part of collagen processing.",
    "Fibroblasts are the cells that synthesize most of the collagen in connective tissue.",
    "Wound healing needs collagen, and for this reason vitamin C is given to patients with poor healing.",
    "The mineralization of bone happens on a collagen scaffold.",
    "Defective collagen cross linking is also seen in lathyrism, which is caused by inhibition of lysyl oxidase.",
]

SLIDES = [
    "Glycolysis\nHexokinase phosphorylates glucose\nPhosphofructokinase regulates glycolysis\nPyruvate kinase produces pyruvate and ATP",
    "Collagen synthesis\nProline hydroxylase needs vitamin C\nScurvy is caused by vitamin C deficiency\nLysyl oxidase needs copper",
]


def segments(sentences: list[str] | None = None, seconds: float = 20.0) -> list[dict]:
    sentences = sentences if sentences is not None else GLYCOLYSIS + COLLAGEN
    return [{"start": i * seconds, "end": (i + 1) * seconds - 0.5, "text": s} for i, s in enumerate(sentences)]


def lecture_index(with_slides: bool = False) -> DocumentIndex:
    return build_index(segments(), SLIDES if with_slides else ())


def write_pdf(path, pages: list[list[str]]):
    """A real text PDF (one page per list of lines) written with pypdf - for tests that read slides."""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    font = writer._add_object(DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                                                NameObject("/BaseFont"): NameObject("/Helvetica")}))
    for lines in pages:
        page = writer.add_blank_page(width=420, height=300)
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        commands = ["BT /F1 12 Tf 20 270 Td 16 TL"]
        for line in lines:
            commands.append("(" + line.replace("(", "").replace(")", "") + ") Tj T*")
        commands.append("ET")
        stream = DecodedStreamObject()
        stream.set_data("\n".join(commands).encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    with open(path, "wb") as handle:
        writer.write(handle)
    return path


def write_lecture_folder(folder, with_clean_text: bool = True):
    """The files of a finished transcription (the ones the study side reads), made from the synthetic lecture."""
    import json
    from pathlib import Path

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    segs = segments()
    (folder / "local_transcript.json").write_text(json.dumps({"language": "en", "segments": segs}), encoding="utf-8")
    text = "\n".join(s["text"] for s in segs)
    (folder / "local_raw_transcript.txt").write_text(text, encoding="utf-8")
    if with_clean_text:
        (folder / "local_clean_transcript.txt").write_text(text, encoding="utf-8")
        (folder / "local_transcript_with_timestamps.txt").write_text(
            "\n".join(f"[{int(s['start'] // 60):02d}:{int(s['start'] % 60):02d}] {s['text']}" for s in segs), encoding="utf-8")
    (folder / "metadata.json").write_text(json.dumps({"language": {"detected": "en"}, "subject": {"subjects": []}}), encoding="utf-8")
    return folder


TURKISH = [
    "Glukozun pirüvata yıkılmasına glikoliz denir.",
    "Glikoliz hücrenin sitoplazmasında gerçekleşir ve iki ATP üretir.",
    "Heksokinaz glukozu glukoz-6-fosfata dönüştürür, bu tepkime glukozu hücre içinde tutar.",
    "Önemli olan nokta şudur: eritrositlerde mitokondri bulunmaz ve bu nedenle yalnızca glikolize bağımlıdırlar.",
    "Pirüvat kinaz eksikliği hemolitik anemiye yol açar çünkü eritrositler ATP için glikolize bağlıdır.",
    "Fosfofruktokinaz glikolizin anahtar düzenleyici enzimidir ve yüksek ATP düzeylerinde baskılanır.",
    "Laktat dehidrogenaz anaerobik koşullarda pirüvatı laktata çevirir ve NAD+ yeniden oluşturur.",
    "Kolajen hücre dışı matriksin en bol bulunan yapısal proteinidir ve üçlü sarmal yapıdadır.",
    "Prolin hidroksilaz için C vitamini gerekir ve C vitamini eksikliği iskorbüte neden olur.",
    "Kolajen çapraz bağlarını lizil oksidaz oluşturur ve bu enzim bakıra ihtiyaç duyar.",
    "Osteogenezis imperfekta kolajen genlerindeki mutasyonlarla ortaya çıkan kalıtsal bir hastalıktır.",
    "Sınavda mutlaka sorulacak: heksokinaz ve glukokinaz arasındaki fark.",
]
