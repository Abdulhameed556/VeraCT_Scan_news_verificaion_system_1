import os
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import requests
import spacy
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import logging
import traceback
import uuid
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()

app = FastAPI()

# Load spaCy model with error handling
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Successfully loaded spaCy model")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {str(e)}")
    logger.error("Attempting to download the model...")
    try:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")
        logger.info("Successfully downloaded and loaded spaCy model")
    except Exception as download_error:
        logger.critical(f"Failed to download spaCy model: {str(download_error)}")
        logger.critical("Continuing without NLP capabilities")
        nlp = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Reliable news sources list (first part shown for brevity)
reliable_sources = [
    # Original Nigerian News Sites
    "punchng.com", "vanguardngr.com", "guardian.ng", "premiumtimesng.com",
    "dailypost.ng", "thenationonlineng.net", "saharareporters.com", "channelstv.com",
    "naijanews.com", "tribuneonlineng.com", "pmnewsnigeria.com", "sunnewsonline.com",
    "leadership.ng", "tvcnews.tv", "thisdaylive.com", "businessday.ng",
    "dailytrust.com", "blueprint.ng", "thecable.ng", "ripplesnigeria.com",
    "newtelegraphng.com", "nigerianbulletin.com", "informationng.com", "ynaija.com",
    "bellanaija.com", "notjustok.com", "naijaloaded.com.ng", "tooxclusive.com",
    "lindaikejisblog.com", "that1960chick.com", "pulse.ng", "olorisupergal.com",
    "fameloaded.com", "nairaland.com",
    
    # Original International News
    "bbc.com", "cnn.com", "aljazeera.com", "reuters.com", "apnews.com",
    "theguardian.com", "nytimes.com", "washingtonpost.com", "bloomberg.com",
    "forbes.com", "cnbc.com", "ft.com", "npr.org", "abcnews.go.com", 
    "nbcnews.com", "latimes.com", "economist.com", "dw.com", "cbc.ca",
    
    # Global Entertainment & Culture
    "eonline.com", "tmz.com", "billboard.com", "variety.com", "hollywoodreporter.com",
    "buzzfeed.com", "people.com", "complex.com", "rollingstone.com", "pitchfork.com",
    "vanityfair.com", "vulture.com", "etonline.com", "mtv.com", "vh1.com",
    
    # Global Sports
    "espn.com", "skysports.com", "goal.com", "bleacherreport.com",
    "eurosport.com", "cbssports.com", "foxsports.com", "nba.com",
    "fifa.com", "uefa.com", "mlssoccer.com", "sportingnews.com",
    
    # Nigerian Sports & Entertainment
    "brila.net", "npfl.ng", "complete-sports.com", "allnigeriasoccer.com",
    
    # African & Pan-African News
    "africanews.com", "mg.co.za", "citinewsroom.com", "theeastafrican.co.ke",
    "ghanaweb.com", "nation.africa", "dailynation.africa", "standardmedia.co.ke",
    "enca.com", "sabcnews.com", "herald.ng", "zambianobserver.com",
    
    # Tech, Business & Startup News
    "techpoint.africa", "techcabal.com", "technext24.com", "benjamindada.com",
    "disrupt-africa.com", "venturesafrica.com", "crunchbase.com", "techcrunch.com",
    "theverge.com", "wired.com", "mashable.com", "thenextweb.com", "hbr.org",
    
    # Finance & Economy
    "cointelegraph.com", "coindesk.com", "investopedia.com", "yahoo.com/finance",
    "marketwatch.com", "nasdaq.com", "barrons.com", "money.cnn.com",
    
    # Education & Science
    "nature.com", "sciencedaily.com", "nationalgeographic.com", "newscientist.com",
    
    # Fact-Checking & Research
    "snopes.com", "factcheck.org", "politifact.com", "fullfact.org",
    
    # ADDITIONAL 500 SOURCES BY REGION AND CATEGORY:
    
    # NORTH AMERICA
    # USA - News & Politics (65)
    "axios.com", "fivethirtyeight.com", "thehill.com", "realclearpolitics.com", "thedailybeast.com",
    "reason.com", "motherjones.com", "theintercept.com", "propublica.org", "rollcall.com",
    "theatlantic.com", "newyorker.com", "harpers.org", "foreignpolicy.com", "foreignaffairs.com",
    "chicagotribune.com", "bostonglobe.com", "denverpost.com", "sfchronicle.com", "dallasnews.com",
    "seattletimes.com", "miamiherald.com", "ajc.com", "startribune.com", "azcentral.com",
    "detroitnews.com", "freep.com", "orlandosentinel.com", "baltimoresun.com", "mercurynews.com",
    "inquirer.com", "houstonchronicle.com", "courier-journal.com", "dispatch.com", "statesman.com",
    "jsonline.com", "tennessean.com", "indystar.com", "sltrib.com", "newsobserver.com",
    "oregonlive.com", "sacbee.com", "star-telegram.com", "reviewjournal.com", "pilotonline.com",
    "buffalonews.com", "arkansasonline.com", "providencejournal.com", "kansascity.com", "desmoinesregister.com",
    "cincinnati.com", "clarionledger.com", "omaha.com", "courier-journal.com", "oklahoman.com",
    "delawareonline.com", "democratandchronicle.com", "greenvilleonline.com", "knoxnews.com", "commercialappeal.com",
    
    # Canada (15)
    "nationalpost.com", "theglobeandmail.com", "macleans.ca", "torontostar.com", "montrealgazette.com",
    "vancouversun.com", "ottawacitizen.com", "calgarysun.com", "edmontonsun.com", "winnipegfreepress.com",
    "thestar.com", "lapresse.ca", "ledevoir.com", "journaldemontreal.com", "tvanouvelles.ca",
    
    # Mexico & Central America (15)
    "eluniversal.com.mx", "milenio.com", "jornada.com.mx", "excelsior.com.mx", "reforma.com",
    "elsalvador.com", "laprensa.hn", "prensalibre.com", "nacion.com", "teletica.com",
    "elsiglo.com.pa", "laprensa.com.ni", "diario.mx", "elnuevodia.com", "periodicocubano.com",
    
    # Caribbean (10)
    "jamaicaobserver.com", "jamaicagleaner.com", "tribune242.com", "thenassauguardian.com", "nationnews.com",
    "trinidadexpress.com", "newsday.co.tt", "barbadostoday.bb", "dominicavibes.dm", "stluciatimes.com",
    
    # SOUTH AMERICA (25)
    "folha.uol.com.br", "globo.com", "estadao.com.br", "clarin.com", "lanacion.com.ar",
    "latercera.com", "emol.com", "eltiempo.com", "elespectador.com", "elcomercio.pe",
    "larepublica.pe", "elobservador.com.uy", "elpais.com.uy", "abc.com.py", "ultimahora.com",
    "eldeber.com.bo", "larazon.com", "eluniverso.com", "elcomercio.com", "lahora.com.ec",
    "eltiempo.com.ve", "elnacional.com", "noticias24.com", "ultimasnoticias.com.ve", "elestimulo.com",
    
    # EUROPE
    # UK & Ireland (20)
    "dailymail.co.uk", "thetimes.co.uk", "thesun.co.uk", "express.co.uk", "standard.co.uk",
    "metro.co.uk", "spectator.co.uk", "newstatesman.com", "channel4.com/news", "itv.com/news",
    "prospectmagazine.co.uk", "theweek.co.uk", "scotsman.com", "heraldscotland.com", "walesonline.co.uk",
    "belfasttelegraph.co.uk", "irishtimes.com", "independent.ie", "rte.ie/news", "thejournal.ie",
    
    # Germany, Austria & Switzerland (20)
    "spiegel.de", "faz.net", "sueddeutsche.de", "zeit.de", "welt.de",
    "tagesschau.de", "stern.de", "handelsblatt.com", "focus.de", "tagesspiegel.de",
    "nzz.ch", "tagesanzeiger.ch", "blick.ch", "20min.ch", "watson.ch",
    "derstandard.at", "diepresse.com", "kurier.at", "orf.at", "krone.at",
    
    # France & Francophone Europe (15)
    "lemonde.fr", "lefigaro.fr", "liberation.fr", "leparisien.fr", "20minutes.fr",
    "lexpress.fr", "lepoint.fr", "nouvelobs.com", "ouest-france.fr", "sudouest.fr",
    "rtbf.be", "lesoir.be", "lalibre.be", "letemps.ch", "tio.ch",
    
    # Italy (10)
    "repubblica.it", "corriere.it", "lastampa.it", "ilsole24ore.com", "ansa.it",
    "rainews.it", "ilfattoquotidiano.it", "ilmessaggero.it", "ilgiornale.it", "adnkronos.com",
    
    # Spain & Portugal (15)
    "elpais.com", "elmundo.es", "abc.es", "lavanguardia.com", "eldiario.es",
    "elconfidencial.com", "20minutos.es", "publico.es", "larazon.es", "rtve.es",
    "publico.pt", "dn.pt", "expresso.pt", "observador.pt", "jn.pt",
    
    # Nordics (15)
    "dn.se", "svd.se", "expressen.se", "aftonbladet.se", "svt.se",
    "dr.dk", "politiken.dk", "berlingske.dk", "tv2.dk", "hs.fi",
    "yle.fi", "vg.no", "aftenposten.no", "nrk.no", "bt.no",
    
    # Eastern Europe (20)
    "gazeta.ru", "kommersant.ru", "rbc.ru", "interfax.ru", "tass.ru",
    "pravda.com.ua", "ukrinform.ua", "kyivpost.com", "unian.info", "wyborcza.pl",
    "onet.pl", "wp.pl", "delfi.lt", "15min.lt", "postimees.ee",
    "24chasa.bg", "novinite.com", "index.hu", "digi24.ro", "hotnews.ro",
    
    # MIDDLE EAST & NORTH AFRICA (40)
    "haaretz.com", "ynetnews.com", "timesofisrael.com", "jpost.com", "alarabiya.net",
    "middleeasteye.net", "al-monitor.com", "dailysabah.com", "hurriyetdailynews.com",
    "ahram.org.eg", "egyptindependent.com", "madamasr.com", "jordantimes.com", "petra.gov.jo",
    "naharnet.com", "lorientlejour.com", "gulfnews.com", "khaleejtimes.com", "thenational.ae",
    "arabnews.com", "saudigazette.com.sa", "timesofoman.com", "thepeninsulaqatar.com", "qatarday.com",
    "bahraintribune.com", "yenisafak.com", "sabq.org", "masrawy.com", "youm7.com",
    "moroccoworldnews.com", "hespress.com", "leconomiste.com", "allafrica.com/morocco", "libyaherald.com",
    "tunisienumerique.com", "leconomistemaghrebin.com", "letemps.com.tn", "algerie-eco.com", "tsa-algerie.com",
    
    # SUB-SAHARAN AFRICA (35)
    # West Africa
    "punchng.com", "thecable.ng", "premiumtimesng.com", "myjoyonline.com", "peacefmonline.com",
    "citifmonline.com", "yen.com.gh", "seneweb.com", "dakaractu.com", "senego.com",
    "apanews.net", "guineenews.org", "abidjan.net", "fratmat.info", "aouaga.com",
    
    # East Africa
    "thecitizen.co.tz", "dailynews.co.tz", "theeastafrican.co.ke", "nation.co.ke", "standardmedia.co.ke",
    "monitor.co.ug", "newvision.co.ug", "newtimes.co.rw", "igihe.com", "ethiopianreporter.com",
    "addisfortune.net", "addisstandard.com", "hiiraan.com", "radiodalsan.com", "shabelle.net",
    
    # Southern Africa
    "sowetanlive.co.za", "dailymaverick.co.za", "businesslive.co.za", "ewn.co.za", "namibian.com.na",
    
    # ASIA
    # East Asia (35)
    "scmp.com", "thestandard.com.hk", "hk01.com", "globaltimes.cn", "chinadaily.com.cn",
    "sixthtone.com", "caixin.com", "yicai.com", "thepaper.cn", "japantimes.co.jp",
    "asahi.com", "mainichi.jp", "yomiuri.co.jp", "nhk.or.jp", "kyodonews.net",
    "koreaherald.com", "koreatimes.co.kr", "chosun.com", "joongang.co.kr", "hani.co.kr",
    "taipeitimes.com", "chinapost.nownews.com", "focustaiwan.tw", "udn.com", "ltn.com.tw",
    "thestar.com.my", "nst.com.my", "malaymail.com", "bernama.com", "thejakartapost.com",
    "kompas.com", "detik.com", "tribunnews.com", "thejakartapost.com", "vinanet.vn",
    
    # South Asia (25)
    "indianexpress.com", "thehindu.com", "hindustantimes.com", "ndtv.com", "news18.com",
    "financialexpress.com", "livemint.com", "telegraphindia.com", "deccanherald.com", "tribuneindia.com",
    "dawn.com", "tribune.com.pk", "geo.tv", "thenews.com.pk", "brecorder.com",
    "dailystar.net", "bdnews24.com", "prothomalo.com", "colombopage.com", "dailymirror.lk",
    "thehimalayantimes.com", "kathmandupost.com", "kuenselonline.com", "bhutantimes.bt", "thebhutanese.bt",
    
    # Central Asia (10)
    "akipress.com", "kabar.kg", "24.kg", "azernews.az", "trend.az",
    "inform.kz", "kazinform.kz", "astanatimes.com", "uzreport.uz", "uza.uz",
    
    # OCEANIA (20)
    "theaustralian.com.au", "afr.com", "theage.com.au", "smh.com.au", "news.com.au",
    "9news.com.au", "abc.net.au", "sbs.com.au", "theguardian.com/au", "watoday.com.au",
    "nzherald.co.nz", "stuff.co.nz", "rnz.co.nz", "tvnz.co.nz", "newshub.co.nz",
    "fijitimes.com", "fijivillage.com", "samoanews.com", "pina.com.fj", "pacnews.org",
    
    # SPECIALIZED MEDIA
    # Technology (25)
    "techcrunch.com", "arstechnica.com", "thenextweb.com", "wired.com", "theverge.com",
    "cnet.com", "zdnet.com", "venturebeat.com", "engadget.com", "gizmodo.com",
    "tomshardware.com", "anandtech.com", "macrumors.com", "androidpolice.com", "xda-developers.com",
    "techradar.com", "pcworld.com", "extremetech.com", "slashdot.org", "bleepingcomputer.com",
    "hackernews.com", "techmeme.com", "technologyreview.com", "hackernoon.com", "9to5mac.com",
    
    # Business & Finance (25)
    "ft.com", "economist.com", "bloomberg.com", "reuters.com", "wsj.com",
    "cnbc.com", "businessinsider.com", "fortune.com", "moneycontrol.com", "livemint.com",
    "marketwatch.com", "seekingalpha.com", "investing.com", "zacks.com", "thestreet.com",
    "morningstar.com", "cnbctv18.com", "barrons.com", "fool.com", "businessstandard.com",
    "ibtimes.com", "hbr.org", "fastcompany.com", "mckinsey.com", "bain.com",
    
    # Science & Health (25)
    "scientificamerican.com", "science.org", "nature.com", "newscientist.com", "livescience.com",
    "sciencedaily.com", "medicalnewstoday.com", "webmd.com", "health.com", "healthline.com",
    "mayoclinic.org", "nih.gov", "sciencealert.com", "medicaldaily.com", "sciencefocus.com",
    "sciencenews.org", "popsci.com", "discovermagazine.com", "phys.org", "sciencemag.org",
    "eurekalert.org", "medscape.com", "thelancet.com", "bmj.com", "nejm.org",
    
    # Sports (25)
    "espn.com", "sports.yahoo.com", "cbssports.com", "nbcsports.com", "foxsports.com",
    "bleacherreport.com", "si.com", "theathletic.com", "sportingnews.com", "sbnation.com",
    "goal.com", "skysports.com", "bbc.com/sport", "eurosport.com", "marca.com",
    "as.com", "lequipe.fr", "kicker.de", "sportbild.de", "sportmediaset.mediaset.it",
    "supersport.com", "sportstarlive.com", "sportal.bg", "gazzetta.gr", "sport24.gr",
    
    # Entertainment (25)
    "variety.com", "hollywoodreporter.com", "deadline.com", "ew.com", "imdb.com",
    "indiewire.com", "screenrant.com", "cinemablend.com", "avclub.com", "empireonline.com",
    "digitalspy.com", "rottentomatoes.com", "metacritic.com", "polygon.com", "kotaku.com",
    "gamespot.com", "ign.com", "eurogamer.net", "rockpapershotgun.com", "pcgamer.com",
    "nintendolife.com", "pushsquare.com", "playstationlifestyle.net", "vg247.com", "gamasutra.com",
    
    # Additional International Fact-Checking (15)
    "factcheck.org", "politifact.com", "snopes.com", "truthorfiction.com", "checkyourfact.com",
    "factcheckni.org", "factnameh.com", "faktograf.hr", "verificat.cat", "maldita.es",
    "correctiv.org", "mimikama.at", "teyit.org", "stopfake.org", "ellinikahoaxes.gr",
    
    # ADDITIONAL CATEGORIES
    # Media Analysis (10)
    "niemanlab.org", "cjr.org", "poynter.org", "mediagazer.com", "mediaite.com",
    "adweek.com", "pressgazette.co.uk", "themediabriefing.com", "mediapost.com", "journalism.co.uk",
    
    # Regional/Local US Media (10)
    "nj.com", "mlive.com", "masslive.com", "pennlive.com", "cleveland.com",
    "al.com", "oregonlive.com", "silive.com", "syracuse.com", "nola.com",
    
    # Travel & Lifestyle (10)
    "travelandleisure.com", "cntraveler.com", "afar.com", "lonelyplanet.com", "fodors.com",
    "timeout.com", "eater.com", "bonappetit.com", "epicurious.com", "seriouseats.com",
    
    # Environmental (10)
    "nationalgeographic.com", "treehugger.com", "grist.org", "ecowatch.com", "ensia.com",
    "earthisland.org", "environmentalhealth.news", "yaleclimateconnections.org", "e360.yale.edu", "insideclimatenews.org"
]

# Load API keys from environment variables with validation
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate API keys
missing_keys = []
if not SEARCH_API_KEY: missing_keys.append("SEARCH_API_KEY")
if not TAVILY_API_KEY: missing_keys.append("TAVILY_API_KEY") 
if not GROQ_API_KEY: missing_keys.append("GROQ_API_KEY")

if missing_keys:
    warning_message = f"Missing API keys: {', '.join(missing_keys)}"
    logger.warning(warning_message)
    if "GROQ_API_KEY" in missing_keys:
        raise ValueError("Missing Groq API key. Set GROQ_API_KEY in your environment.")

# LangChain Groq + LLaMA model
llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    groq_api_key=GROQ_API_KEY
)

class NewsVerificationRequest(BaseModel):
    headline: str
    content: str
    source_url: str = Field(default="", description="Optional source URL for direct verification")
    
    # Add validators to ensure non-empty strings
    @validator('headline', 'content')
    def must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Field cannot be empty or whitespace')
        return v.strip()
    
    @validator('source_url')
    def validate_url(cls, v):
        if v and v.strip():
            # Simple URL validation
            url_pattern = re.compile(
                r'^(?:http|https)://'  # http:// or https://
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
                r'localhost|'  # localhost
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or ip
                r'(?::\d+)?'  # optional port
                r'(?:/?|[/?]\S+)$', re.IGNORECASE)
            
            if not url_pattern.match(v.strip()):
                raise ValueError('Invalid URL format')
            return v.strip()
        return ""

# Enhanced helper function for better domain matching
def is_from_reliable_source(url):
    try:
        if not url:
            logger.debug("Empty URL provided to is_from_reliable_source")
            return False
            
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Remove 'www.' prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
            
        # Debug output
        logger.debug(f"Checking domain: {domain}")
        
        # Extract base domain for better matching
        domain_parts = domain.split('.')
        if len(domain_parts) > 2:
            base_domain = '.'.join(domain_parts[-2:])
        else:
            base_domain = domain
            
        # New extended domain checking logic
        for source in reliable_sources:
            # Direct match
            if domain == source:
                logger.debug(f"Direct match found for {domain}")
                return True
                
            # Subdomain match
            if domain.endswith(f".{source}"):
                logger.debug(f"Subdomain match found for {domain} with source {source}")
                return True
                
            # Path-based match (for instances like medium.com/reliable-publisher)
            if source.endswith(f".{domain}"):
                logger.debug(f"Path-based match found for {domain} with source {source}")
                return True
                
            # Base domain match
            if base_domain == source:
                logger.debug(f"Base domain match found for {domain}")
                return True
                
        logger.debug(f"No reliable source match found for {domain}")
        return False
    except Exception as e:
        logger.error(f"Error parsing URL {url}: {str(e)}")
        return False

@app.get("/test-apis")
async def test_apis():
    """Endpoint to test if API keys are working"""
    results = {}
    
    # Test SearchAPI
    if SEARCH_API_KEY:
        try:
            search_api_url = "https://www.searchapi.io/api/v1/search?q=test&engine=google_news"
            search_headers = {"Authorization": f"Bearer {SEARCH_API_KEY}"}
            response = requests.get(search_api_url, headers=search_headers, timeout=5)
            results["search_api"] = {
                "status": "working" if response.status_code == 200 else "error",
                "status_code": response.status_code
            }
        except Exception as e:
            results["search_api"] = {"status": "error", "message": str(e)}
    else:
        results["search_api"] = {"status": "missing_key"}
    
    # Test Tavily API
    if TAVILY_API_KEY:
        try:
            tavily_url = "https://api.tavily.com/search"
            tavily_payload = {
                "api_key": TAVILY_API_KEY,
                "query": "test",
                "search_depth": "basic",
                "num_results": 1
            }
            response = requests.post(tavily_url, json=tavily_payload, timeout=5)
            results["tavily_api"] = {
                "status": "working" if response.status_code == 200 else "error",
                "status_code": response.status_code
            }
        except Exception as e:
            results["tavily_api"] = {"status": "error", "message": str(e)}
    else:
        results["tavily_api"] = {"status": "missing_key"}
    
    # Test Groq API
    if GROQ_API_KEY:
        try:
            # Simple test using LangChain's ChatGroq
            test_prompt = ChatPromptTemplate.from_template("Say 'Hello, API test'")
            chain = test_prompt | llm
            response = chain.invoke({})
            results["groq_api"] = {
                "status": "working" if "hello" in response.content.lower() else "error",
                "response": response.content[:30] + "..." if len(response.content) > 30 else response.content
            }
        except Exception as e:
            results["groq_api"] = {"status": "error", "message": str(e)}
    else:
        results["groq_api"] = {"status": "missing_key"}
    
    return results

# New function to test domain matching
@app.get("/test-domain")
async def test_domain(url: str):
    """Endpoint to test domain matching logic"""
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Remove 'www.' prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
            
        # Extract base domain for better matching
        domain_parts = domain.split('.')
        if len(domain_parts) > 2:
            base_domain = '.'.join(domain_parts[-2:])
        else:
            base_domain = domain
        
        matching_sources = []
        for source in reliable_sources:
            if (domain == source or 
                domain.endswith(f".{source}") or 
                source.endswith(f".{domain}") or
                base_domain == source):
                matching_sources.append(source)
        
        is_reliable = len(matching_sources) > 0
        
        return {
            "url": url,
            "parsed_domain": domain,
            "base_domain": base_domain,
            "is_reliable_source": is_reliable,
            "matching_sources": matching_sources,
            "reliable_sources_checked": len(reliable_sources)
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error analyzing domain: {str(e)}"
        )

@app.post("/verify")
async def verify_news(news: NewsVerificationRequest):
    try:
        # Create a session ID for tracking this verification request
        session_id = f"session-{uuid.uuid4().hex[:8]}"
        logger.info(f"[{session_id}] Received verification request at {datetime.utcnow().isoformat()}")
        
        # Initialize variables
        found_in_reliable_sources = False
        direct_source_verified = False
        potential_match = False
        fake_news = None
        final_verdict = None
        current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        current_user = "NewsVerifierSystem"
        api_failures = []
        search_results_count = 0
        matched_source_urls = []
        
        # Generate verification ID
        verification_id = f"{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"
        
        # Validate input data
        if not news.headline or not news.headline.strip():
            logger.error(f"[{session_id}] Empty headline provided")
            raise HTTPException(
                status_code=400,
                detail="Headline cannot be empty"
            )
        
        if not news.content or not news.content.strip():
            logger.error(f"[{session_id}] Empty content provided")
            raise HTTPException(
                status_code=400,
                detail="Content cannot be empty"
            )

        # Log the input data (sanitized)
        logger.info(f"[{session_id}] Processing headline: {news.headline[:50]}...")
        logger.info(f"[{session_id}] Processing content length: {len(news.content)} characters")
        if news.source_url:
            logger.info(f"[{session_id}] Source URL provided: {news.source_url}")
            
        # Clean and normalize input
        news.headline = news.headline.strip()
        news.content = news.content.strip()

        # DIRECT SOURCE URL VERIFICATION (New feature)
        if news.source_url:
            logger.info(f"[{session_id}] Checking provided source URL: {news.source_url}")
            if is_from_reliable_source(news.source_url):
                logger.info(f"[{session_id}] Source URL directly verified as reliable: {news.source_url}")
                direct_source_verified = True
                found_in_reliable_sources = True
                potential_match = True
                matched_source_urls.append(news.source_url)
            else:
                logger.warning(f"[{session_id}] Provided source URL not in reliable sources list: {news.source_url}")

        # Extract named entities with error handling
        entities = ""
        if nlp:
            try:
                doc = nlp(news.headline + " " + news.content[:1000])  # Limit content length for entity extraction
                entities = " ".join([ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PERSON", "LOC", "DATE", "EVENT"]])
                
                logger.info(f"[{session_id}] Extracted entities: {entities[:100]}...")
            
                if not entities:
                    logger.warning(f"[{session_id}] No entities found in the text")
                    entities = news.headline  # Fall back to using the headline for searching
            except Exception as e:
                logger.error(f"[{session_id}] Entity extraction error: {str(e)}")
                logger.error(traceback.format_exc())
                entities = news.headline  # Fall back to using the headline for searching
        else:
            # If NLP is not available, use the headline
            entities = news.headline
            logger.warning(f"[{session_id}] Using headline for search due to unavailable NLP")

        # Only perform API searches if direct source verification failed
        if not direct_source_verified:
            # SearchAPI Lookup with improved error handling
            search_results = {"organic_results": []}
            if SEARCH_API_KEY:
                try:
                    # Construct a better search query with keywords from headline and entities
                    search_query = f"{news.headline} {entities[:100]}"
                    search_api_url = f"https://www.searchapi.io/api/v1/search?q={search_query}&engine=google_news"
                    search_headers = {"Authorization": f"Bearer {SEARCH_API_KEY}"}
                    
                    logger.info(f"[{session_id}] Initiating SearchAPI request with query: {search_query[:50]}...")
                    response_search = requests.get(search_api_url, headers=search_headers, timeout=15)
                    
                    if not response_search.ok:
                        logger.warning(f"[{session_id}] SearchAPI returned status code: {response_search.status_code}")
                        api_failures.append(f"SearchAPI (Status {response_search.status_code})")
                    else:
                        search_results = response_search.json()
                        search_results_count += len(search_results.get("organic_results", []))
                        logger.info(f"[{session_id}] SearchAPI returned {search_results_count} results")

                    # Check search results against reliable sources with improved matching
                    for result in search_results.get("organic_results", []):
                        result_url = result.get("link", "")
                        if is_from_reliable_source(result_url):
                            found_in_reliable_sources = True
                            potential_match = True
                            matched_source_urls.append(result_url)
                            logger.info(f"[{session_id}] Found match in reliable source: {result_url}")

                except requests.exceptions.RequestException as e:
                    logger.error(f"[{session_id}] SearchAPI error: {str(e)}")
                    api_failures.append("SearchAPI")
            else:
                logger.warning(f"[{session_id}] SearchAPI key not available, skipping this search provider")
                api_failures.append("SearchAPI (No API Key)")

            # Tavily Lookup with improved error handling
            tavily_results = {"results": []}
            if TAVILY_API_KEY:
                try:
                    tavily_url = "https://api.tavily.com/search"
                    tavily_payload = {
                        "api_key": TAVILY_API_KEY,
                        "query": f"{news.headline} {entities[:100]}",  # Use headline and entities for better results
                        "search_depth": "advanced",  # Use advanced depth for better results
                        "include_domains": reliable_sources[:50],  # Include more reliable sources
                        "num_results": 15
                    }
                    
                    logger.info(f"[{session_id}] Initiating Tavily API request")
                    response_tavily = requests.post(tavily_url, json=tavily_payload, timeout=15)
                    
                    if not response_tavily.ok:
                        logger.warning(f"[{session_id}] Tavily API returned status code: {response_tavily.status_code}")
                        api_failures.append(f"Tavily API (Status {response_tavily.status_code})")
                    else:
                        tavily_results = response_tavily.json()
                        search_results_count += len(tavily_results.get("results", []))
                        logger.info(f"[{session_id}] Tavily API returned {len(tavily_results.get('results', []))} results")

                    # Check Tavily results against reliable sources with improved matching
                    for result in tavily_results.get("results", []):
                        result_url = result.get("url", "")
                        if is_from_reliable_source(result_url):
                            found_in_reliable_sources = True
                            potential_match = True
                            matched_source_urls.append(result_url)
                            logger.info(f"[{session_id}] Found match in Tavily results: {result_url}")

                except requests.exceptions.RequestException as e:
                    logger.error(f"[{session_id}] Tavily API error: {str(e)}")
                    api_failures.append("Tavily API")
            else:
                logger.warning(f"[{session_id}] Tavily API key not available, skipping this search provider")
                api_failures.append("Tavily API (No API Key)")

        # Log source verification results
        logger.info(f"[{session_id}] Found in reliable sources: {found_in_reliable_sources}")
        logger.info(f"[{session_id}] Direct source verified: {direct_source_verified}")
        logger.info(f"[{session_id}] Potential match: {potential_match}")
        logger.info(f"[{session_id}] Search results count: {search_results_count}")
        logger.info(f"[{session_id}] Matched reliable sources: {matched_source_urls}")
        
        # Prepare source verification context
        source_verification_note = ""
        if api_failures and not direct_source_verified:
            source_verification_note = f"Note: Some verification services were unavailable: {', '.join(api_failures)}."
        
        # Source confidence scoring - NEW
        source_confidence = 0
        if direct_source_verified:
            source_confidence = 90  # High confidence for direct source verification
            logger.info(f"[{session_id}] Source confidence: 90 (Direct verification)")
        elif found_in_reliable_sources:
            # Scale confidence based on number of matches
            source_confidence = min(85, 50 + (len(matched_source_urls) * 10))
            logger.info(f"[{session_id}] Source confidence: {source_confidence} ({len(matched_source_urls)} matches)")
        elif search_results_count > 0:
            # Some results but no reliable sources
            source_confidence = 30
            logger.info(f"[{session_id}] Source confidence: 30 (No reliable matches)")
        else:
            # No results at all
            source_confidence = 10
            logger.info(f"[{session_id}] Source confidence: 10 (No search results)")

        # LLM Final Verdict with improved prompt
        try:
            # Current date for context
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            prompt = ChatPromptTemplate.from_template("""
            You are an advanced news verification system analyzing the following content:
            
            Headline: {headline}
            Content: {content}
            
            Current date: {current_date}
            
            Additional Context:
            - Found in reliable sources: {found_in_sources}
            - Direct source verification: {direct_verified}
            - Source confidence score: {source_confidence}/100
            - Entity matches found: {has_matches}
            - Search results found: {search_count}
            - Matched reliable sources: {matched_sources}
            {source_note}
            
            Please analyze with extreme care, especially for claims about:
            - Deaths or injuries
            - Major events or disasters
            - Political statements
            - Financial markets
            - Public health information
            
            IMPORTANT INSTRUCTIONS:
            1. DO NOT flag news as fake solely based on dates, as some articles may legitimately reference future events or dates.
            2. Focus on factual inconsistencies, logical contradictions, evidence verification.
            3. If source verification succeeded (direct or indirect), give strong weight to this positive signal.
            4. Consider the nature and specificity of claims made in the content.
            5. If the news is from a verified reliable source, and there are no clear contradictions or implausibilities, lean towards verification.
            
            Provide a detailed analysis followed by your final verdict.
            
            Your verdict must be ONE of these exact phrases on its own line:
            VERDICT: VERIFIED (Use if high confidence in authenticity, especially with reliable source confirmation)
            VERDICT: LIKELY FAKE (Use only if clear evidence of fabrication)
            VERDICT: REQUIRES MORE VERIFICATION (Use for uncertain cases)
            """)
            
            logger.info(f"[{session_id}] Initiating LLM analysis")
            chain = prompt | llm
            llm_response = chain.invoke({
                "headline": news.headline,
                "content": news.content,
                "found_in_sources": "Yes" if found_in_reliable_sources else "No",
                "direct_verified": "Yes" if direct_source_verified else "No",
                "source_confidence": source_confidence,
                "has_matches": "Yes" if potential_match else "No",
                "search_count": search_results_count,
                "matched_sources": ", ".join(matched_source_urls[:3]) if matched_source_urls else "None",
                "source_note": source_verification_note,
                "current_date": current_date
            })
            logger.info(f"[{session_id}] LLM analysis completed")
            
        except Exception as e:
            logger.error(f"[{session_id}] LLM analysis error: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Error during LLM analysis: {str(e)}"
            )

        # Parse LLM response
        llm_response_upper = llm_response.content.upper()
        
        # Define verdict indicators
        fake_indicators = [
            "LIKELY FAKE", "FALSE", "NOT RELIABLE", "UNVERIFIED",
            "MISINFORMATION", "NO EVIDENCE", "CANNOT VERIFY",
            "SUSPICIOUS", "MISLEADING", "FABRICATED"
        ]
        
        verify_indicators = [
            "VERIFIED", "RELIABLE", "CONFIRMED", "ACCURATE",
            "AUTHENTIC", "LEGITIMATE", "TRUSTWORTHY", "FACTUAL"
        ]
        
        # Calculate indicator scores
        fake_score = sum(1 for indicator in fake_indicators if indicator in llm_response_upper)
        verify_score = sum(1 for indicator in verify_indicators if indicator in llm_response_upper)

        # Determine verdict with improved logic - give more weight to source verification
        if "VERDICT: VERIFIED" in llm_response_upper:
            # Direct source verification is strong evidence
            if direct_source_verified or (found_in_reliable_sources and source_confidence >= 60) or verify_score > 2:
                final_verdict = "Verified"
                fake_news = False
                logger.info(f"[{session_id}] Content verified as authentic")
            else:
                final_verdict = "Requires More Verification"
                fake_news = None
                logger.info(f"[{session_id}] Content marked for further verification - LLM verdict positive but weak source confidence")
        elif "VERDICT: LIKELY FAKE" in llm_response_upper:
            # If directly verified source but LLM says fake, require further verification
            if direct_source_verified:
                final_verdict = "Requires More Verification"
                fake_news = None
                logger.info(f"[{session_id}] Content marked for further verification - conflict between source verification and LLM analysis")
            else:
                final_verdict = "Likely Fake"
                fake_news = True
                logger.info(f"[{session_id}] Content flagged as likely fake")
        else:
            final_verdict = "Requires More Verification"
            fake_news = None
            logger.info(f"[{session_id}] Content requires additional verification")

        # Create status message based on verdict
        if fake_news is False:
            status_message = f"""✅ VERIFIED | Confidence Level: High

VERIFICATION REPORT ID: VR-{verification_id}

This news has been analyzed by our verification system and appears to be authentic.
{"The content is from a directly verified reliable source." if direct_source_verified else "The content has been cross-referenced with reliable sources."} 
It passes our fact-checking criteria.

Verification completed: {current_time}
"""
        elif fake_news is True:
            status_message = f"""⚠️ LIKELY FAKE | Alert Level: High

ALERT REPORT ID: FR-{verification_id}

Our verification system has detected potential reliability issues with this content.
Exercise caution before sharing or acting on this information.

Verification completed: {current_time}
"""
        else:
            status_message = f"""🔄 VERIFICATION PENDING | Status: In Progress

ASSESSMENT ID: PR-{verification_id}

This content requires additional verification. Our system could not definitively 
determine its authenticity based on available information.

Preliminary assessment completed: {current_time}
"""

        # Enhanced verification metrics
        verification_metrics = {
            "source_validation_score": source_confidence,
            "content_reliability_index": 85 if fake_news is False else 30 if fake_news is True else 50,
            "entity_verification_status": "Validated" if fake_news is False else "Failed" if fake_news is True else "Pending",
            "ai_confidence_level": "High" if direct_source_verified or source_confidence >= 70 or abs(verify_score - fake_score) > 2 else "Medium",
            "analyst": current_user,
            "verification_id": verification_id,
            "api_failures": api_failures,
            "search_results_found": search_results_count,
            "direct_source_verified": direct_source_verified,
            "matched_reliable_sources": len(matched_source_urls)
        }

        return {
            "verdict": final_verdict,
            "status_message": status_message,
            "llm_response": llm_response.content,
            "verification_data": {
                "direct_source_verified": direct_source_verified,
                "found_in_reliable_sources": found_in_reliable_sources,
                "potential_match": potential_match,
                "source_confidence": source_confidence,
                "fake_indicators_found": fake_score,
                "verify_indicators_found": verify_score,
                "matched_sources": matched_source_urls[:5] if matched_source_urls else [],
                "verification_metrics": verification_metrics,
                "timestamp_utc": current_time,
                "analyst": current_user,
                "report_id": f"{'VR' if fake_news is False else 'FR' if fake_news is True else 'PR'}-{verification_id}"
            }
        }

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting NewsVerifier API")
    uvicorn.run(app, host="0.0.0.0", port=8000)