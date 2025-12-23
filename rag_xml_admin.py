
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

st.set_page_config(page_title="PAA RAG Flight Admin", layout="wide")

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

    "OZ": "Asiana Airlines",

    "ABA": "AIRBORNE AVIATION",

    "AJ": "Air Belgium International",

    "PA": "Airblue",

    "RU": "AirBridgeCargo",

    "G9": "Air Arabia",

    "AC": "Air Canada",

    "UX": "Air Europa",

    "YP": "Aero Lloyd",

    "FG": "Ariana Afghan Air",

    "SU": "Aeroflot Russian Airlines",

    "AF": "Air France",

    "J2": "Azerbaijan Airlines",

    "AI": "Air India",

    "IZ": "Arkia-Israeli Airlines Ltd",

    "AK": "Turkmenistan Airline",

    "UL": "Srilankan Airlines",

    "DP": "Air 2000 Ltd",

    "TZ": "American Trans Air",

    "NH": "Ana All Nippon Airways",

    "KA": "Aero Nomad Airlines",

    "YW": "Air Nostrum",

    "NZ": "Air New Zealand",

    "IW": "Aom French Airlines",

    "AP": "PAK ARAB FERTILIZER",

    "AR": "Aerolineas Argentinas",

    "6Y": "SmartLynx Airlines",

    "ASS": "AIRCRAFT SALES AND SERVICES (P",

    "OS": "Austrian Airlines",

    "AV": "Avianca",

    "YK": "Avia",

    "ZT": "TITAN AIRWAYS LTD",

    "AD": "Aero World",

    "M4": "Avioimpex",

    "9X": "New Axis Airways",

    "AO": "Aviaco",

    "AZ": "Alitalia ",

    "7L": "Silkway West",

    "ZP": "Silk Way Airline",

    "ZR": "Aviacon Zitotrans",

    "UM": "Air Zimbabwe",

    "7L": "Air Bristol",

    "BY": "Britannia Airways Ltd",

    "BA": "British Airways",

    "BG": "Biman Bangladesh Airl.",

    "BD": "Bmi British Midland",

    "JA": "Air Bosna",

    "BV": "Blu-Express",

    "BU": "Braathens Asa",

    "E5": "Samara Airlines",

    "V9": "Bashkir Airlines",

    "BT": "BAHRIA TOWN",

    "DB": "Brit Air",

    "PV": "CIVIL AVIATION ",

    "CAF": "China Airfore",

    "CA": "Air China",

    "CP": "Canadian Airlines",

    "MU": "China Eastern Airlines",

    "IV": "Fujian Airlines",

    "CF": "Zhongfei General Aviation",

    "G8": "Air Great Wall",

    "HU": "Hainan Airlines",

    "ZJ": "Cnac Zhejiang Airlines",

    "CK": "China Cargo ",

    "CV": "Cargolux Airlines Internationa",

    "OY": "UN",

    "CMS": "CHIEF MINISTER GOVT. OF SINDH",

    "CO": "Continental Airlines",

    "SS": "Corse Air Int.",

    "3U": "Sichuanair",

    "CZ": "China Southern Airline",

    "O3": "SF Airline",

    "ZH": "Shenzhen Airlines",

    "OU": "Croatia Airlines",

    "KN": "China United Airlines",

    "X5": "Cronus Airlines",

    "Z2": "Airasia Zest",

    "8Y": "China Postal Airlines",

    "DAF": "Danish Airforce",

    "9J": "Dana Air",

    "DR": "Deer Air",

    "HO": "Antinea Airlines",

    "LH": "Deutsche Lufthansa",

    "D9": "Donavia",

    "DPP": "DEPARTMENT OF PLANT PROTECTION",

    "DX": "Danish Air Transport",

    "EZ": "Evergreen International Airlin",

    "ES": "Estonian Air",

    "LY": "El Al Israel Airlines",

    "E4": "Enter Air",

    "ESM": "ETIHAD SUGAR MILLS LIMITED",

    "EY": "Etihad Airways",

    "ET": "Ethiopian Airlines",

    "EV": "Air Evex",

    "LI": "ETF Airways",

    "IF": "Fly Baghdad",

    "FZ": "Fly Dubai",

    "FH": "Free Bird",

    "AY": "Finnair",

    "9P": "Fly Jinnah",

    "AL": "Aero Leasing",

    "GE": "Global Aviation",

    "G2": "Gullivair",

    "D4": "GEO SKY",

    "DA": "Air Georgia",

    "GF": "Gulf Air",

    "GA": "Garuda Indonesia",

    "9C": "Gill Airways",

    "GLY": "GALAXY AVIATION (PVT) LTD",

    "ST": "Germania Fluggesellschaft",

    "GOP": "VIP Flight GOVT OF PUNJAB",

    "5Y": "Atlas Air",

    "TD": "Air Anatolia",

    "GW": "GULF WINGS",

    "5M": "Hi Fly Malta",

    "5K": "Hi Fly",

    "H9": "Himalya Airline",

    "HSP": "HOME SECRETARY GOVT OF PUNJAB",

    "VN": "Vietnam Airlines",

    "HYB": "HYBRID TECHNICS",

    "YG": "CARGO AIRLINE (YTO)",

    "IA": "Iraqi Airways",

    "IB": "Iberia",

    "IBH": "IBRAHIM HOLDINGS (Pvt.) Ltd.",

    "FI": "Icelandair",

    "ICN": "ICON AIR (PVT)LTD.",

    "F3": "Red Angel",

    "IGB": "Balochistan Police",

    "IR": "Iran Air",

    "B9": "Islamic Republic of Iran",

    "W5": "Mahan Air",

    "IL": "Istanbul Airlines",

    "TY": "Iberworld",

    "IY": "Yemenia Yemen Airways",

    "JI": "Jade Cargo International",

    "JL": "Japan Airlines",

    "JD": "Japan Air System",

    "JU": "Air Serbia",

    "R5": "Jordan Aviation",

    "JW": "JDW Aviation (Pvt) Ltd",

    "JY": "Jersey European Airways",

    "JK": "Spanair",

    "JSP": "PEREGRINE AVIATION (PVT) LTD",

    "JZR": "Jazeera Airways",

    "J9": "Jazeera Airyways",

    "KU": "Kuwait Airways",

    "KE": "Korean Air",

    "KFC": "KARACHI AERO CLUB LTD.",

    "GW": "Kuban Airlines",

    "KL": "Royal Dutch Airlines",

    "RQ": "Kam Air",

    "KNC": "KING CRETE ASSOCIATE",

    "XY": "Nas Air",

    "KTA": "K2 AIRWAYS",

    "KT": "Kibris Turkish Airlines",

    "K4": "Kazakhstan Airlines",

    "9Y": "Air Kazakstan",

    "LA": "Lan Chile",

    "QV": "Lao Airlines",

    "LZ": "Balkan Bulgarian Airlines",

    "BJ": "Nouvelair Tunisie",

    "LCH": "Lake City",

    "LCY": "LUCKY CEMENT LIMITED",

    "H7": "Air Alfa Hava Yollari",

    "LFC": "LAHORE FLYING CLUB",

    "LO": "Polskie Linie Lotnicze Lot",

    "LT": "Ltu International Airways",

    "GV": "Leisure Int. Airways",

    "IN": "Mat Macedonian Airlines",

    "MV": "Air Mediterranean",

    "MH": "Malaysia Airlines",

    "MK": "Air Mauritius",

    "AE": "Mandarin Airlines Ltd.",

    "ME": "Midle East",

    "MFC": "MULTAN FLYING CLUB",

    "OM": "Miat-Mongolian Airlines",

    "LQ": "Lanmei Airlines",

    "ZB": "Monarch Airlines",

    "MP": "Martin Air",

    "I6": "Air Indus",

    "MSA": "MAKHDUM SYED AHMED MAHMUD.",

    "SM": "Air Cairo",

    "M9": "Motor Sich Airlines",

    "MS": "Egyptair",

    "6M": "Maximus Air Cargo",

    "N1": "Gulf Stream",

    "N8": "National Airlines",

    "NT": "Nishat Airline",

    "X9": "Avion Express",

    "OX": "Orient Thai Airlines",

    "8Q": "Onur Air Tasimacilik",

    "BK": "Okay Airways",

    "OL": "Olympus Airways",

    "WY": "Oman Ar",

    "OML": "OMNI AVIATION (Pvt.)Ltd.",

    "OV": "Salam Air",

    "PES": "Chitral Airport",

    "R2": "Orenburg Airlines",

    "XW": "Orbit Express Airlines",

    "PO": "Polar Air Cargo",

    "PAG": "AGHA KHAN FOUNDATION",

    "PR": "Philippine Airlines",

    "PAV": "PAK AVIATION & AVIATORS LAHORE",

    "9Q": "Pb Air",

    "Q8": "Pacific East Asia Cargo",

    "PFC": "PESHAWAR FLYING CLUB",

    "NI": "Portugalia Airlines",

    "PHO": "PHOENIX AVIATION SERVICES",

    "PK": "Pakistan International Airline",

    "PK1": "PRESIDENT AIRCRAFT",

    "PK2": "PRIME MINISTER AIRCRAFT",

    "Z8": "Pulkovo Aviation Enterpr.",

    "EB": "Wamos Air",

    "J5": "Sochi Airlines Aviaprima",

    "PS": "Pak Services Limited",

    "P6": "Privilege Style",

    "FP": "Fly Pro",

    "QF": "Qantas Airways",

    "QE": "Qatar Executive",

    "AS": "African Safari",

    "QR": "Qatar Airways",

    "BI": "Royal Brunei Airlines",

    "MC": "US-MILITARY",

    "RFC": "RAWALPINDI FLYING CLUB",

    "VM": "Regional Airlines",

    "RJ": "Royal Jordanian",

    "RT": "Rak Airways",

    "R3": "Armenian Airlines",

    "WQ": "Romavia",

    "RA": "Nepal Airlines",

    "ROJ": "ROYALJET",

    "RO": "Tarom ",

    "QN": "Royal Aviation Inc",

    "RR": "General Aviation",

    "ZS": "Safair",

    "FR": "Ryanair Ltd",

    "SA": "South African Airways",

    "NL": "Shaheen Air",

    "SK": "Sas Scandinavian Airlines",

    "SCA": "KK AVIATION LIMITED",

    "6E": "Malmoe Aviation",

    "ER": "Serene Air",

    "2R": "Star Airlines",

    "HM": "Air Seychelles Ltd.",

    "4Q": "Safi Airways",

    "6S": "Saudigulf Airlines",

    "SHA": "SYED SIBGHATULLAH SHAH",

    "SH": "SHARJAH",

    "SQ": "Singapore Airlines",

    "SIE": "DEWAN AVIATION (PVT) LTD",

    "PF": "AIR SIAL",

    "Q7": "Sobelair",

    "S3": "San Air Company",

    "DM": "Anda Air",

    "STA": "STAR AVIATION",

    "SV": "Saudi Airline",

    "8S": "Saak Stavropol Airlines",

    "SW": "SWF",

    "SWL": "SKY WINGS (PVT) LIMITED",

    "LX": "Swiss International Airlines",

    "TP": "Air Portugal Tap",

    "I3": "ATA Airlines",

    "TG": "Thai Airline",

    "WE": "Thai Smile Airline",

    "TK": "Turkish Airlines",

    "9I": "Thai Sky Airlines",

    "T7": "Transaer",

    "SH": "Air Toulouse",

    "B6": "Top Air",

    "FF": "Tower Air",

    "3T": "Tarco Airline",

    "TS": "Air Transat",

    "UN": "Transaero Airlines",

    "BH": "Tea Switzerland (Basel Ag)",

    "T5": "Turkmenistan Airlines Cgo",

    "QS": "Smartwings",

    "TQ": "Braathens Sverige Ab",

    "EX": "Thai Express Air",

    "7J": "Tajikair",

    "N6": "TCA Airline",

    "EK": "Emirates Airline",

    "UA": "United Airlines",

    "UB": "Myanmar Airways",

    "Z6": "Dniproavia",

    "UEP": "UNITED ENERGY PAKISTAN LTD C/O",

    "UFC": "Z4 ULTRA LIGHT SPORTS AND RECR",

    "6U": "Air Ukraine",

    "US": "Us Airways",

    "HY": "Uzbekistan Airways",

    "VC": "Ocean Airlines",

    "VEN": "Venus Pakistan (Pvt) Limited",

    "VL": "Air Via Bulgarian Airways",

    "VS": "Virgin Atlantic Airways",

    "VIS": "Vision Airline ",

    "FV": "Viva Air",

    "VT": "Vista Jet",

    "VG": "Vlm Vlaamse Luchttranspo.",

    "9V": "Vip-Air",

    "RG": "Varig Airlines",

    "VP": "Vasp",

    "8K": "Air Ostrava",

    "PK": "Pakistan International Airline",

    "Y8": "Yangzi River Cargo",

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
