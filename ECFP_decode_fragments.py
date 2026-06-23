"""
Décode les top 20 bits ECFP4 les plus impactants pour HOMO, LUMO, GAP.
Pour chaque bit : retrouve l'enchaînement d'atomes correspondant via RDKit,
et vérifie les doublons dans le parquet.
"""
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Draw, AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

RADIUS = 4
TOP_N  = 20
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS)

# ── Chargement ────────────────────────────────────────────────────────────────
df = pd.read_parquet("ecfp4_top1000_dataset.parquet")
print(f"Parquet chargé : {df.shape[0]:,} molécules x {df.shape[1]:,} colonnes\n")

ecfp_cols = [c for c in df.columns if c.startswith("ECFP4_")]

# ── Vérif doublons de bit_id ──────────────────────────────────────────────────
bit_ids = [int(c.replace("ECFP4_", "")) for c in ecfp_cols]
counts  = Counter(bit_ids)
dups    = {k: v for k, v in counts.items() if v > 1}
if dups:
    print(f"⚠  {len(dups)} bit_id(s) présent(s) EN DOUBLE dans les colonnes :")
    for bid, n in dups.items():
        print(f"   ECFP4_{bid}  →  {n} colonnes")
else:
    print("✓  Aucun doublon de bit_id dans les colonnes du parquet.")

# ── Régression Ridge → top 20 par cible ──────────────────────────────────────
mask = df[["HOMO", "LUMO", "GAP"]].notna().all(axis=1)
df_c = df[mask & (df["HOMO"] >= -15) & (df["LUMO"] >= -15)].reset_index(drop=True)
X    = df_c[ecfp_cols].values.astype(np.float32)

print(f"\nMolécules utilisées : {len(df_c):,}\n")

top20_by_target = {}
for target in ["HOMO", "LUMO", "GAP"]:
    y = df_c[target].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = Ridge(alpha=1.0)
    m.fit(Xtr, ytr)
    order = np.argsort(np.abs(m.coef_))[::-1][:TOP_N]
    top20_by_target[target] = [(ecfp_cols[j], m.coef_[j]) for j in order]

# ── Collecte des bit_ids uniques à décoder ────────────────────────────────────
all_top_bits = set()
for cols_coefs in top20_by_target.values():
    for col, _ in cols_coefs:
        all_top_bits.add(int(col.replace("ECFP4_", "")))

print(f"Bits uniques à décoder (union HOMO+LUMO+GAP) : {len(all_top_bits)}\n")

# ── Décodage : pour chaque bit_id, trouver UN exemple de sous-structure ──────
def decode_bit(bit_id, df_c, ecfp_cols, radius=RADIUS, max_mol=3000):
    """
    Retourne (smiles_mol, atom_idx_central) d'un exemple portant ce bit_id.
    On utilise GetSparseCountFingerprint + GetBitInfo (atom mapping via
    GetMorganGenerator avec includeRedundantEnvironments=False).
    """
    col = f"ECFP4_{bit_id}"
    if col not in df_c.columns:
        return None, None, None

    # Molécules qui possèdent ce bit (count >= 1)
    subset = df_c[df_c[col] >= 1]["SMILES"]
    if len(subset) == 0:
        return None, None, None

    gen_info = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, includeRedundantEnvironments=False
    )

    for smiles in subset.head(max_mol):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        ao = {}   # atom output
        _ = gen_info.GetSparseCountFingerprint(mol, fromAtoms=None,
                                               ignoreAtoms=None,
                                               atomInvariants=None,
                                               bondInvariants=None,
                                               additionalOutput=ao)
        bit_info = ao.get("atomCounts", {})
        # Approche alternative : GetMorganBitInfo via GetCountFingerprint
        # On utilise rdkit legacy pour GetBitInfo
        fp_info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=4096,
                                                    bitInfo=fp_info)
        # On cherche via sparse fingerprint
        fp_sparse = AllChem.GetMorganFingerprint(mol, radius)
        nz = fp_sparse.GetNonzeroElements()
        if bit_id in nz:
            # Trouver l'atome central via GetMorganBitInfo sur version non hashed
            env_info = {}
            AllChem.GetMorganFingerprint(mol, radius, bitInfo=env_info)
            if bit_id in env_info:
                central_atom, rad = env_info[bit_id][0]
                return smiles, mol, central_atom
    return None, None, None


def get_env_smiles(mol, center_atom, radius):
    """Extrait le fragment SMILES centré sur center_atom à distance radius."""
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_atom)
    if not env:
        return Chem.MolToSmiles(mol.GetAtomWithIdx(center_atom).GetOwningMol()), []
    amap = {}
    submol = Chem.PathToSubmol(mol, env, atomMap=amap)
    rev_amap = {v: k for k, v in amap.items()}
    atom_symbols = []
    for new_idx in sorted(rev_amap):
        orig_idx = rev_amap[new_idx]
        sym = mol.GetAtomWithIdx(orig_idx).GetSymbol()
        atom_symbols.append(f"{sym}(idx={orig_idx})")
    return Chem.MolToSmiles(submol), atom_symbols


# ── Résultats ─────────────────────────────────────────────────────────────────
decoded = {}
print("Décodage des sous-structures ...")
for bit_id in sorted(all_top_bits):
    smiles_mol, mol, center = decode_bit(bit_id, df_c, ecfp_cols)
    if mol is None:
        decoded[bit_id] = {"frag_smiles": "INTROUVABLE", "atoms": []}
        continue
    frag_smiles, atoms = get_env_smiles(mol, center, RADIUS // 2)
    decoded[bit_id] = {
        "mol_smiles":  smiles_mol,
        "center_atom": center,
        "atom_symbol": mol.GetAtomWithIdx(center).GetSymbol(),
        "frag_smiles":  frag_smiles,
        "atoms_detail": atoms,
    }

# ── Affichage par cible ───────────────────────────────────────────────────────
for target in ["HOMO", "LUMO", "GAP"]:
    print(f"\n{'═'*70}")
    print(f"  Top {TOP_N} bits Ridge — {target}")
    print(f"{'═'*70}")
    print(f"  {'Rang':<5} {'Bit ID':<14} {'Coef':>8}  {'Atome central':<6}  Fragment SMILES")
    print(f"  {'-'*65}")
    for rank, (col, coef) in enumerate(top20_by_target[target], 1):
        bid   = int(col.replace("ECFP4_", ""))
        d     = decoded.get(bid, {})
        sym   = d.get("atom_symbol", "?")
        frag  = d.get("frag_smiles", "?")
        atoms = d.get("atoms_detail", [])
        print(f"  {rank:<5} {bid:<14} {coef:>+8.4f}  {sym:<6}  {frag}")
        if atoms:
            print(f"        Atomes: {', '.join(atoms[:8])}")

# ── Tableau récapitulatif exporté ─────────────────────────────────────────────
rows = []
for target in ["HOMO", "LUMO", "GAP"]:
    for rank, (col, coef) in enumerate(top20_by_target[target], 1):
        bid = int(col.replace("ECFP4_", ""))
        d   = decoded.get(bid, {})
        rows.append({
            "target":       target,
            "rank":         rank,
            "ecfp4_id":     bid,
            "ridge_coef":   round(coef, 5),
            "center_atom":  d.get("atom_symbol", "?"),
            "frag_smiles":  d.get("frag_smiles", "?"),
            "example_mol":  d.get("mol_smiles", "?"),
        })

recap = pd.DataFrame(rows)
recap.to_csv("ecfp4_top20_decoded.csv", index=False)
print(f"\n\nTableau sauvegardé → ecfp4_top20_decoded.csv")
print(recap.to_string(index=False))
