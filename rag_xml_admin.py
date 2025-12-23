import os
import re
import csv
import json
import glob
import xml.etree.ElementTree as ET
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from sentence_transformers import SentenceTransformer
import streamlit as st

# ================= PAGE CONFIG =================
st.set_page_config(page_title="PAA XML Flight Data RAG Admin", layout="wide", page_icon="✈️")
st.title("✈️ PAA RAG Flight Admin - CSV & XML Ingestion")

# ================= CONFIG =================
DATA_FOLDER = "rag_xml_data"
WEAVIATE_URL = st.secrets["WEAVIATE_URL"]
WEAVIATE_KEY = st.secrets["WEAVIATE_API_KEY"]
COLLECTION_NAME = "PAA_XML_FLIGHTS"


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
EMBED = load_embedder()

# ================= MAPPINGS =================
# Flight Nature
FLIGHT_NATURE_DESC = {
    "CGO": "Cargo Flight",
    "GEN": "General Flight",
    "PAX": "Passenger Flight",
    "SPE": "Special Flight",
    "VIP": "VIP Flight",
}

# Flight Sector
FLIGHT_SECTOR_DESC = {
    "D": "Domestic",
    "I": "International"
}

# Flight Status
FLIGHT_STATUS_DESC = {
    "AB":"Airborne","AD":"Advanced Flight","BD":"Boarding","CC":"Check-in close",
    "CI":"Check-in open","CN":"Confirmed","CX":"Cancelled","DV":"Diverted","ES":"Estimated",
    "EX":"Expected","FB":"First Bag","FS":"Final Approach","FX":"Flight Fixed","GA":"Gate Attended",
    "GC":"Gate Closed","GO":"Gate Open","LB":"Last Bag","LC":"Last Call","LD":"Landed","NI":"Next Information",
    "NO":"Non-operational","NT":"New Time","OB":"On/Off Blocks","OT":"On Time","OV":"Overshoot","RS":"Return To Stand",
    "SH":"Scheduled","XF":"Flight Fixed","ZN":"Zoning"
}


# Airline ICAO->IATA->Name (sample, full list to be inserted)
AIRLINE_ICAO_TO_IATA = {
    "AAP": "AAP",
    "AAR": "OZ",
    "ABA": "ABA",
    "ABB": "AJ",
    "ABQ": "PA",
    "ABW": "RU",
    "ABY": "G9",
    "ACA": "AC",
    "AEA": "UX",
    "AEF": "YP",
    "AFG": "FG",
    "AFL": "SU",
    "AFR": "AF",
    "AHY": "J2",
    "AIC": "AI",
    "AIZ": "IZ",
    "AKH": "AK",
    "ALK": "UL",
    "AMM": "DP",
    "AMT": "TZ",
    "ANA": "NH",
    "ANK": "KA",
    "ANS": "YW",
    "ANZ": "NZ",
    "AOM": "IW",
    "APF": "AP",
    "ARG": "AR",
    "ART": "6Y",
    "ASS": "ASS",
    "AUA": "OS",
    "AVA": "AV",
    "AVJ": "YK",
    "AWC": "ZT",
    "AWP": "AD",
    "AXX": "M4",
    "AXY": "9X",
    "AYC": "AO",
    "AZA": "AZ",
    "AZG": "7L",
    "AZQ": "ZP",
    "AZS": "ZR",
    "AZW": "UM",
    "AZX": "7L",
    "BAL": "BY",
    "BAW": "BA",
    "BBC": "BG",
    "BMA": "BD",
    "BON": "JA",
    "BPA": "BV",
    "BRA": "BU",
    "BRZ": "E5",
    "BTC": "V9",
    "BTP": "BT",
    "BZH": "DB",
    "CAA": "PV",
    "CAF": "CAF",
    "CCA": "CA",
    "CDN": "CP",
    "CES": "MU",
    "CFJ": "IV",
    "CFZ": "CF",
    "CGW": "G8",
    "CHH": "HU",
    "CJG": "ZJ",
    "CKK": "CK",
    "CLX": "CV",
    "CMB": "OY",
    "CMS": "CMS",
    "COA": "CO",
    "CRL": "SS",
    "CSC": "3U",
    "CSN": "CZ",
    "CSS": "O3",
    "CSZ": "ZH",
    "CTN": "OU",
    "CUA": "KN",
    "CUS": "X5",
    "CYN": "Z2",
    "CYZ": "8Y",
    "DAF": "DAF",
    "DAN": "9J",
    "DER": "DR",
    "DJA": "HO",
    "DLH": "LH",
    "DNV": "D9",
    "DPP": "DPP",
    "DTR": "DX",
    "EIA": "EZ",
    "ELL": "ES",
    "ELY": "LY",
    "ENT": "E4",
    "ESM": "ESM",
    "ETD": "EY",
    "ETH": "ET",
    "EVE": "EV",
    "EZZ": "LI",
    "FBA": "IF",
    "FDB": "FZ",
    "FHY": "FH",
    "FIN": "AY",
    "FJL": "9P",
    "FPG": "AL",
    "GBB": "GE",
    "GBG": "G2",
    "GEL": "D4",
    "GEO": "DA",
    "GFA": "GF",
    "GIA": "GA",
    "GIL": "9C",
    "GLY": "GLY",
    "GMI": "ST",
    "GOP": "GOP",
    "GTI": "5Y",
    "GTK": "TD",
    "GWC": "GW",
    "HFM": "5M",
    "HFY": "5K",
    "HIM": "H9",
    "HSP": "HSP",
    "HVN": "VN",
    "HYB": "HYB",
    "HYT": "YG",
    "IAW": "IA",
    "IBE": "IB",
    "IBH": "IBH",
    "ICE": "FI",
    "ICN": "ICN",
    "IFA": "F3",
    "IGB": "IGB",
    "IRA": "IR",
    "IRB": "B9",
    "IRM": "W5",
    "IST": "IL",
    "IWD": "TY",
    "IYE": "IY",
    "JAE": "JI",
    "JAL": "JL",
    "JAS": "JD",
    "JAT": "JU",
    "JAV": "R5",
    "JDW": "JW",
    "JEA": "JY",
    "JKK": "JK",
    "JSP": "JSP",
    "JZE": "JZR",
    "JZR": "J9",
    "KAC": "KU",
    "KAL": "KE",
    "KFC": "KFC",
    "KIL": "GW",
    "KLM": "KL",
    "KMF": "RQ",
    "KNC": "KNC",
    "KNE": "XY",
    "KTA": "KTA",
    "KYV": "KT",
    "KZA": "K4",
    "KZK": "9Y",
    "LAN": "LA",
    "LAO": "QV",
    "LAZ": "LZ",
    "LBT": "BJ",
    "LCH": "LCH",
    "LCY": "LCY",
    "LFA": "H7",
    "LFC": "LFC",
    "LOT": "LO",
    "LTU": "LT",
    "LUI": "GV",
    "MAK": "IN",
    "MAR": "MV",
    "MAS": "MH",
    "MAU": "MK",
    "MDA": "AE",
    "MEA": "ME",
    "MFC": "MFC",
    "MGL": "OM",
    "MKR": "LQ",
    "MON": "ZB",
    "MPH": "MP",
    "MPK": "I6",
    "MSA": "MSA",
    "MSC": "SM",
    "MSI": "M9",
    "MSR": "MS",
    "MXM": "6M",
    "N19": "N1",
    "NCR": "N8",
    "NST": "NT",
    "NVD": "X9",
    "OEA": "OX",
    "OHY": "8Q",
    "OKA": "BK",
    "OLY": "OL",
    "OMA": "WY",
    "OML": "OML",
    "OMS": "OV",
    "OPC": "PES",
    "ORB": "R2",
    "ORX": "XW",
    "PAC": "PO",
    "PAG": "PAG",
    "PAL": "PR",
    "PAV": "PAV",
    "PBA": "9Q",
    "PEC": "Q8",
    "PFC": "PFC",
    "PGA": "NI",
    "PHO": "PHO",
    "PIA": "PK",
    "PK1": "PK1",
    "PK2": "PK2",
    "PLK": "Z8",
    "PLM": "EB",
    "PRL": "J5",
    "PSL": "PS",
    "PVG": "P6",
    "PVV": "FP",
    "QFA": "QF",
    "QQE": "QE",
    "QSC": "AS",
    "QTR": "QR",
    "RBA": "BI",
    "RCH": "MC",
    "RFC": "RFC",
    "RGI": "VM",
    "RJA": "RJ",
    "RKM": "RT",
    "RME": "R3",
    "RMV": "WQ",
    "RNA": "RA",
    "ROJ": "ROJ",
    "ROT": "RO",
    "ROY": "QN",
    "RRR": "RR",
    "RSF": "ZS",
    "RYR": "FR",
    "SAA": "SA",
    "SAI": "NL",
    "SAS": "SK",
    "SCA": "SCA",
    "SCW": "6E",
    "SEP": "ER",
    "SEU": "2R",
    "SEY": "HM",
    "SFW": "4Q",
    "SGQ": "6S",
    "SHA": "SHA",
    "SHJ": "SH",
    "SIA": "SQ",
    "SIE": "SIE",
    "SIF": "PF",
    "SLR": "Q7",
    "SND": "S3",
    "SSV": "DM",
    "STA": "STA",
    "SVA": "SV",
    "SVL": "8S",
    "SWF": "SW",
    "SWL": "SWL",
    "SWR": "LX",
    "TAP": "TP",
    "TBZ": "I3",
    "THA": "TG",
    "THD": "WE",
    "THY": "TK",
    "TKY": "9I",
    "TLA": "T7",
    "TLE": "SH",
    "TOP": "B6",
    "TOW": "FF",
    "TQQ": "3T",
    "TSC": "TS",
    "TSO": "UN",
    "TSW": "BH",
    "TUA": "T5",
    "TVS": "QS",
    "TWE": "TQ",
    "TXZ": "EX",
    "TZK": "7J",
    "TZS": "N6",
    "UAE": "EK",
    "UAL": "UA",
    "UBA": "UB",
    "UDN": "Z6",
    "UEP": "UEP",
    "UFC": "UFC",
    "UKR": "6U",
    "USA": "US",
    "UZB": "HY",
    "VCX": "VC",
    "VEN": "VEN",
    "VIM": "VL",
    "VIR": "VS",
    "VIS": "VIS",
    "VIV": "FV",
    "VJT": "VT",
    "VLM": "VG",
    "VPA": "9V",
    "VRG": "RG",
    "VSP": "VP",
    "VTR": "8K",
    "YYY": "PK",
    "YZR": "Y8",
    "ZZZ": "ZZZ",
}



AIRLINE_ICAO_TO_NAME = {

      "AAP": "Air Academy Pakistan",
    "AAR": "Asiana Airlines",
    "ABA": "AIRBORNE AVIATION",
    "ABB": "Air Belgium International",
    "ABQ": "Airblue",
    "ABW": "AirBridgeCargo",
    "ABY": "Air Arabia",
    "ACA": "Air Canada",
    "AEA": "Air Europa",
    "AEF": "Aero Lloyd",
    "AFG": "Ariana Afghan Air",
    "AFL": "Aeroflot Russian Airlines",
    "AFR": "Air France",
    "AHY": "Azerbaijan Airlines",
    "AIC": "Air India",
    "AIZ": "Arkia-Israeli Airlines Ltd",
    "AKH": "Turkmenistan Airline",
    "ALK": "Srilankan Airlines",
    "AMM": "Air 2000 Ltd",
    "AMT": "American Trans Air",
    "ANA": "Ana All Nippon Airways",
    "ANK": "Aero Nomad Airlines",
    "ANS": "Air Nostrum",
    "ANZ": "Air New Zealand",
    "AOM": "Aom French Airlines",
    "APF": "PAK ARAB FERTILIZER",
    "ARG": "Aerolineas Argentinas",
    "ART": "SmartLynx Airlines",
    "ASS": "AIRCRAFT SALES AND SERVICES (P",
    "AUA": "Austrian Airlines",
    "AVA": "Avianca",
    "AVJ": "Avia",
    "AWC": "TITAN AIRWAYS LTD",
    "AWP": "Aero World",
    "AXX": "Avioimpex",
    "AXY": "New Axis Airways",
    "AYC": "Aviaco",
    "AZA": "Alitalia ",
    "AZG": "Silkway West",
    "AZQ": "Silk Way Airline",
    "AZS": "Aviacon Zitotrans",
    "AZW": "Air Zimbabwe",
    "AZX": "Air Bristol",
    "BAL": "Britannia Airways Ltd",
    "BAW": "British Airways",
    "BBC": "Biman Bangladesh Airl.",
    "BMA": "Bmi British Midland",
    "BON": "Air Bosna",
    "BPA": "Blu-Express",
    "BRA": "Braathens Asa",
    "BRZ": "Samara Airlines",
    "BTC": "Bashkir Airlines",
    "BTP": "BAHRIA TOWN",
    "BZH": "Brit Air",
    "CAA": "CIVIL AVIATION ",
    "CAF": "China Airfore",
    "CCA": "Air China",
    "CDN": "Canadian Airlines",
    "CES": "China Eastern Airlines",
    "CFJ": "Fujian Airlines",
    "CFZ": "Zhongfei General Aviation",
    "CGW": "Air Great Wall",
    "CHH": "Hainan Airlines",
    "CJG": "Cnac Zhejiang Airlines",
    "CKK": "China Cargo ",
    "CLX": "Cargolux Airlines Internationa",
    "CMB": "UN",
    "CMS": "CHIEF MINISTER GOVT. OF SINDH",
    "COA": "Continental Airlines",
    "CRL": "Corse Air Int.",
    "CSC": "Sichuanair",
    "CSN": "China Southern Airline",
    "CSS": "SF Airline",
    "CSZ": "Shenzhen Airlines",
    "CTN": "Croatia Airlines",
    "CUA": "China United Airlines",
    "CUS": "Cronus Airlines",
    "CYN": "Airasia Zest",
    "CYZ": "China Postal Airlines",
    "DAF": "Danish Airforce",
    "DAN": "Dana Air",
    "DER": "Deer Air",
    "DJA": "Antinea Airlines",
    "DLH": "Deutsche Lufthansa",
    "DNV": "Donavia",
    "DPP": "DEPARTMENT OF PLANT PROTECTION",
    "DTR": "Danish Air Transport",
    "EIA": "Evergreen International Airlin",
    "ELL": "Estonian Air",
    "ELY": "El Al Israel Airlines",
    "ENT": "Enter Air",
    "ESM": "ETIHAD SUGAR MILLS LIMITED",
    "ETD": "Etihad Airways",
    "ETH": "Ethiopian Airlines",
    "EVE": "Air Evex",
    "EZZ": "ETF Airways",
    "FBA": "Fly Baghdad",
    "FDB": "Fly Dubai",
    "FHY": "Free Bird",
    "FIN": "Finnair",
    "FJL": "Fly Jinnah",
    "FPG": "Aero Leasing",
    "GBB": "Global Aviation",
    "GBG": "Gullivair",
    "GEL": "GEO SKY",
    "GEO": "Air Georgia",
    "GFA": "Gulf Air",
    "GIA": "Garuda Indonesia",
    "GIL": "Gill Airways",
    "GLY": "GALAXY AVIATION (PVT) LTD",
    "GMI": "Germania Fluggesellschaft",
    "GOP": "VIP Flight GOVT OF PUNJAB",
    "GTI": "Atlas Air",
    "GTK": "Air Anatolia",
    "GWC": "GULF WINGS",
    "HFM": "Hi Fly Malta",
    "HFY": "Hi Fly",
    "HIM": "Himalya Airline",
    "HSP": "HOME SECRETARY GOVT OF PUNJAB",
    "HVN": "Vietnam Airlines",
    "HYB": "HYBRID TECHNICS",
    "HYT": "CARGO AIRLINE (YTO)",
    "IAW": "Iraqi Airways",
    "IBE": "Iberia",
    "IBH": "IBRAHIM HOLDINGS (Pvt.) Ltd.",
    "ICE": "Icelandair",
    "ICN": "ICON AIR (PVT)LTD.",
    "IFA": "Red Angel",
    "IGB": "Balochistan Police",
    "IRA": "Iran Air",
    "IRB": "Islamic Republic of Iran",
    "IRM": "Mahan Air",
    "IST": "Istanbul Airlines",
    "IWD": "Iberworld",
    "IYE": "Yemenia Yemen Airways",
    "JAE": "Jade Cargo International",
    "JAL": "Japan Airlines",
    "JAS": "Japan Air System",
    "JAT": "Air Serbia",
    "JAV": "Jordan Aviation",
    "JDW": "JDW Aviation (Pvt) Ltd",
    "JEA": "Jersey European Airways",
    "JKK": "Spanair",
    "JSP": "PEREGRINE AVIATION (PVT) LTD",
    "JZE": "Jazeera Airways",
    "JZR": "Jazeera Airyways",
    "KAC": "Kuwait Airways",
    "KAL": "Korean Air",
    "KFC": "KARACHI AERO CLUB LTD.",
    "KIL": "Kuban Airlines",
    "KLM": "Royal Dutch Airlines",
    "KMF": "Kam Air",
    "KNC": "KING CRETE ASSOCIATE",
    "KNE": "Nas Air",
    "KTA": "K2 AIRWAYS",
    "KYV": "Kibris Turkish Airlines",
    "KZA": "Kazakhstan Airlines",
    "KZK": "Air Kazakstan",
    "LAN": "Lan Chile",
    "LAO": "Lao Airlines",
    "LAZ": "Balkan Bulgarian Airlines",
    "LBT": "Nouvelair Tunisie",
    "LCH": "Lake City",
    "LCY": "LUCKY CEMENT LIMITED",
    "LFA": "Air Alfa Hava Yollari",
    "LFC": "LAHORE FLYING CLUB",
    "LOT": "Polskie Linie Lotnicze Lot",
    "LTU": "Ltu International Airways",
    "LUI": "Leisure Int. Airways",
    "MAK": "Mat Macedonian Airlines",
    "MAR": "Air Mediterranean",
    "MAS": "Malaysia Airlines",
    "MAU": "Air Mauritius",
    "MDA": "Mandarin Airlines Ltd.",
    "MEA": "Midle East",
    "MFC": "MULTAN FLYING CLUB",
    "MGL": "Miat-Mongolian Airlines",
    "MKR": "Lanmei Airlines",
    "MON": "Monarch Airlines",
    "MPH": "Martin Air",
    "MPK": "Air Indus",
    "MSA": "MAKHDUM SYED AHMED MAHMUD.",
    "MSC": "Air Cairo",
    "MSI": "Motor Sich Airlines",
    "MSR": "Egyptair",
    "MXM": "Maximus Air Cargo",
    "N19": "Gulf Stream",
    "NCR": "National Airlines",
    "NST": "Nishat Airline",
    "NVD": "Avion Express",
    "OEA": "Orient Thai Airlines",
    "OHY": "Onur Air Tasimacilik",
    "OKA": "Okay Airways",
    "OLY": "Olympus Airways",
    "OMA": "Oman Ar",
    "OML": "OMNI AVIATION (Pvt.)Ltd.",
    "OMS": "Salam Air",
    "OPC": "Chitral Airport",
    "ORB": "Orenburg Airlines",
    "ORX": "Orbit Express Airlines",
    "PAC": "Polar Air Cargo",
    "PAG": "AGHA KHAN FOUNDATION",
    "PAL": "Philippine Airlines",
    "PAV": "PAK AVIATION & AVIATORS LAHORE",
    "PBA": "Pb Air",
    "PEC": "Pacific East Asia Cargo",
    "PFC": "PESHAWAR FLYING CLUB",
    "PGA": "Portugalia Airlines",
    "PHO": "PHOENIX AVIATION SERVICES",
    "PIA": "Pakistan International Airline",
    "PK1": "PRESIDENT AIRCRAFT",
    "PK2": "PRIME MINISTER AIRCRAFT",
    "PLK": "Pulkovo Aviation Enterpr.",
    "PLM": "Wamos Air",
    "PRL": "Sochi Airlines Aviaprima",
    "PSL": "Pak Services Limited",
    "PVG": "Privilege Style",
    "PVV": "Fly Pro",
    "QFA": "Qantas Airways",
    "QQE": "Qatar Executive",
    "QSC": "African Safari",
    "QTR": "Qatar Airways",
    "RBA": "Royal Brunei Airlines",
    "RCH": "US-MILITARY",
    "RFC": "RAWALPINDI FLYING CLUB",
    "RGI": "Regional Airlines",
    "RJA": "Royal Jordanian",
    "RKM": "Rak Airways",
    "RME": "Armenian Airlines",
    "RMV": "Romavia",
    "RNA": "Nepal Airlines",
    "ROJ": "ROYALJET",
    "ROT": "Tarom ",
    "ROY": "Royal Aviation Inc",
    "RRR": "General Aviation",
    "RSF": "Safair",
    "RYR": "Ryanair Ltd",
    "SAA": "South African Airways",
    "SAI": "Shaheen Air",
    "SAS": "Sas Scandinavian Airlines",
    "SCA": "KK AVIATION LIMITED",
    "SCW": "Malmoe Aviation",
    "SEP": "Serene Air",
    "SEU": "Star Airlines",
    "SEY": "Air Seychelles Ltd.",
    "SFW": "Safi Airways",
    "SGQ": "Saudigulf Airlines",
    "SHA": "SYED SIBGHATULLAH SHAH",
    "SHJ": "SHARJAH",
    "SIA": "Singapore Airlines",
    "SIE": "DEWAN AVIATION (PVT) LTD",
    "SIF": "AIR SIAL",
    "SLR": "Sobelair",
    "SND": "San Air Company",
    "SSV": "Anda Air",
    "STA": "STAR AVIATION",
    "SVA": "Saudi Airline",
    "SVL": "Saak Stavropol Airlines",
    "SWF": "SWF",
    "SWL": "SKY WINGS (PVT) LIMITED",
    "SWR": "Swiss International Airlines",
    "TAP": "Air Portugal Tap",
    "TBZ": "ATA Airlines",
    "THA": "Thai Airline",
    "THD": "Thai Smile Airline",
    "THY": "Turkish Airlines",
    "TKY": "Thai Sky Airlines",
    "TLA": "Transaer",
    "TLE": "Air Toulouse",
    "TOP": "Top Air",
    "TOW": "Tower Air",
    "TQQ": "Tarco Airline",
    "TSC": "Air Transat",
    "TSO": "Transaero Airlines",
    "TSW": "Tea Switzerland (Basel Ag)",
    "TUA": "Turkmenistan Airlines Cgo",
    "TVS": "Smartwings",
    "TWE": "Braathens Sverige Ab",
    "TXZ": "Thai Express Air",
    "TZK": "Tajikair",
    "TZS": "TCA Airline",
    "UAE": "Emirates Airline",
    "UAL": "United Airlines",
    "UBA": "Myanmar Airways",
    "UDN": "Dniproavia",
    "UEP": "UNITED ENERGY PAKISTAN LTD C/O",
    "UFC": "Z4 ULTRA LIGHT SPORTS AND RECR",
    "UKR": "Air Ukraine",
    "USA": "Us Airways",
    "UZB": "Uzbekistan Airways",
    "VCX": "Ocean Airlines",
    "VEN": "Venus Pakistan (Pvt) Limited",
    "VIM": "Air Via Bulgarian Airways",
    "VIR": "Virgin Atlantic Airways",
    "VIS": "Vision Airline ",
    "VIV": "Viva Air",
    "VJT": "Vista Jet",
    "VLM": "Vlm Vlaamse Luchttranspo.",
    "VPA": "Vip-Air",
    "VRG": "Varig Airlines",
    "VSP": "Vasp",
    "VTR": "Air Ostrava",
    "YYY": "Pakistan International Airline",
    "YZR": "Yangzi River Cargo",
    "ZZZ": "General Aviation",
}


# ================= HELPERS =================
def clean_text(raw):
    return re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", "", raw).strip()

def parse_checkin_desk_range(range_str):
    # Example: "02-09-02-15" => {"zone":2, "start":9, "end":15}
    parts = range_str.split("-")
    if len(parts) == 4:
        return {
            "zone_start": int(parts[0]),
            "counter_start": int(parts[1]),
            "zone_end": int(parts[2]),
            "counter_end": int(parts[3])
        }
    return {}

def parse_csv_file(path):
    records = []
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_row = {k.strip(): v.strip() for k, v in row.items() if v.strip()}
            if clean_row:
                records.append(clean_row)
    return records

def parse_xml_file(path):
    records = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = clean_text(f.read())
        envelopes = re.findall(r'(<Envelope[\s\S]*?</Envelope>)', raw)
        ns = {"ns": "http://schema.ultra-as.com"}
        
        for env_xml in envelopes:
            try:
                root = ET.fromstring(env_xml)
            except ET.ParseError:
                continue
                
            flight_data = root.find(".//ns:AFDSFlightData", ns)
            if not flight_data:
                continue

            flight_ident = flight_data.find(".//ns:FlightIdentification", ns)
            if not flight_ident:
                continue

            flight_id = flight_ident.findtext("ns:FlightIdentity", default=None, namespaces=ns)
            direction = flight_ident.findtext("ns:FlightDirection", default=None, namespaces=ns)
            sched_date = flight_ident.findtext("ns:ScheduledDate", default=None, namespaces=ns)

            fd = flight_data.find(".//ns:FlightData", ns)
            airport = fd.find(".//ns:Airport", ns) if fd is not None else None
            flight = fd.find(".//ns:Flight", ns) if fd is not None else None
            ops = fd.find(".//ns:OperationalTimes", ns) if fd is not None else None

            carrier_icao = flight.findtext("ns:CarrierICAOCode", default=None, namespaces=ns) if flight else None
            carrier_iata = AIRLINE_ICAO_TO_IATA.get(carrier_icao, flight.findtext("ns:CarrierIATACode", default=None, namespaces=ns) if flight else None)
            carrier_name = AIRLINE_ICAO_TO_NAME.get(carrier_icao, "")


            flight_nature_code = flight.findtext("ns:FlightNatureCode", default=None, namespaces=ns) if flight else None
            flight_sector_code = flight.findtext("ns:FlightSectorCode", default=None, namespaces=ns) if flight else None
            flight_status_code = flight.findtext("ns:FlightStatusCode", default=None, namespaces=ns) if flight else None

            checkin_range = flight.findtext("ns:CheckinDeskRange", default=None, namespaces=ns) if flight else None
            parsed_checkin = parse_checkin_desk_range(checkin_range) if checkin_range else {}


            record = {
                "flight_number": flight_id,
                "direction": direction,
                "scheduled_date": sched_date,
                "carrier_icao": carrier_icao,
                "carrier_iata": carrier_iata,
                "carrier_name": carrier_name,
                "airport": airport.findtext("ns:AirportIATACode", default=None, namespaces=ns) if airport else None,
                "flight_nature_code": flight_nature_code,
                "flight_nature_desc": FLIGHT_NATURE_DESC.get(flight_nature_code, flight_nature_code),
                "flight_sector_code": flight_sector_code,
                "flight_sector_desc": FLIGHT_SECTOR_DESC.get(flight_sector_code, flight_sector_code),
                "flight_status_code": flight_status_code,
                "flight_status_desc": FLIGHT_STATUS_DESC.get(flight_status_code, flight_status_code),
                "scheduled_time": ops.findtext("ns:ScheduledDateTime", default=None, namespaces=ns) if ops else None,
                "actual_time": ops.findtext("ns:LatestKnownDateTime", default=None, namespaces=ns) if ops else None,
                "port_of_call_iata": flight.findtext("ns:PortOfCallIATACode", default=None, namespaces=ns) if flight else None,
                "port_of_call_icao": flight.findtext("ns:PortOfCallICAOCode", default=None, namespaces=ns) if flight else None,
                "checkin_open": flight.findtext("ns:CheckinOpenDateTime", default=None, namespaces=ns) if flight is not None else None,
                "checkin_close": flight.findtext("ns:CheckinCloseDateTime", default=None, namespaces=ns) if flight is not None else None,
                "checkin_desk_range": parsed_checkin,
                "checkin_type": flight.findtext("ns:CheckinTypeCode", default=None, namespaces=ns) if flight is not None else None,
                "gate_open": airport.findtext("ns:GateOpenDateTime", default=None, namespaces=ns) if airport is not None else None,
                "gate_close": airport.findtext("ns:GateCloseDateTime", default=None, namespaces=ns) if airport is not None else None,
                "gate_number": airport.findtext("ns:GateNumber", default=None, namespaces=ns) if airport is not None else None,
                "stand_position": airport.findtext("ns:StandPosition", default=None, namespaces=ns) if airport is not None else None,
                "handling_agent": flight.findtext("ns:HandlingAgentIATACode", default=None, namespaces=ns) if flight is not None else None,
            }
            records.append(record)
    return records

# ================= WEAVIATE =================

def ingest_to_weaviate(records):
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_KEY)
    )

    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)

    coll = client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="flight_number", data_type=DataType.TEXT),
            Property(name="direction", data_type=DataType.TEXT),
            Property(name="airport", data_type=DataType.TEXT),
            Property(name="gate_number", data_type=DataType.TEXT),
            Property(name="flight_status_desc", data_type=DataType.TEXT),
            Property(name="scheduled_time", data_type=DataType.TEXT),
            Property(name="summary", data_type=DataType.TEXT),
        ]
    )

    with coll.batch.dynamic() as batch:
        for f in records:
            summary = (
                f"Flight {f.get('flight_number')} ({f.get('carrier_name')}) is a {f.get('direction')} flight. "
                f"Nature: {f.get('flight_nature_desc')}, Sector: {f.get('flight_sector_desc')}, "
                f"Status: {f.get('flight_status_desc')}. "
                f"Airport: {f.get('airport')}, Gate: {f.get('gate_number')}. "
                f"Scheduled: {f.get('scheduled_time')}, Latest Known: {f.get('actual_time')}."
            )
            batch.add_object(
                properties={
                    "flight_number": f.get("flight_number"),
                    "direction": f.get("direction"),
                    "airport": f.get("airport"),
                    "gate_number": f.get("gate_number"),
                    "flight_status_desc": f.get("flight_status_desc"),
                    "scheduled_time": f.get("scheduled_time"),
                    "summary": summary
                },
                vector=EMBED.encode(summary).tolist()
            )
    client.close()

# ================= UI =================

st.markdown(f"### 📂 Scanning folder: {DATA_FOLDER}")
files = glob.glob(os.path.join(DATA_FOLDER, "*.*"))

if not files:
    st.error("No CSV or XML files found in the data folder.")
    st.stop()

st.write(files)

if st.button("🏗️ Parse & Index All Files"):
    all_records = []
    for f in files:
        if f.lower().endswith(".csv"):
            all_records.extend(parse_csv_file(f))
        elif f.lower().endswith(".xml"):
            all_records.extend(parse_xml_file(f))

    if all_records:
        ingest_to_weaviate(all_records)
        st.success(f"✅ Indexed {len(all_records)} flight records successfully!")
        st.balloons()
    else:
        st.warning("No valid flight records found.")
