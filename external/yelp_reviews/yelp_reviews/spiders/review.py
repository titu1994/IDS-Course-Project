import scrapy
import unicodedata

# all_restaurant_names = ['The Dearborn', 'Stock and Ledger', 'Remingtons', 'Roanoke Restaurant', 'The Gage', 'The Marq',
#                         'MingHin Cuisine', 'Nandos PERiPERi', 'Cochon Volant Brasserie', 'Green Apple PHOever',
#                         'Ge Pa De Caffe', 'Roti Modern Mediterranean', 'Shake Shack', 'Pastoral', 'Seven Lions',
#                         'Wildberry Pancakes and Cafe', 'Heaven on Seven Wabash', 'Millers Pub',
#                         'American Craft Kitchen Bar', 'Oasis Cafe', 'Filini', '312 Chicago', 'Brown Bag Seafood',
#                         'Union Squared', 'Osaka Sushi Express Fresh Fruit Smoothies', 'Billy Goat Tavern Lake Street',
#                         'Eggys Diner', 'Exchequer Restaurant Pub', 'Protein Bar', 'Tokyo Lunch Boxes', 'Atwood',
#                         'Elephant Castle', 'La Bamba', 'Park Grill at Millennium Park', 'Early Society',
#                         'University Club of Chicago', 'BenjYehuda', 'Spotted Monkey', 'Frontera Fresco',
#                         'Prime Provisions',
#                         'Lukes Lobster City Hall', 'Acanto', 'Brightwok Kitchen', 'Trattoria No 10', 'Naf Naf Grill',
#                         'Cindys', 'Broken English Taco Pub The Loop', 'Revival Food Hall', 'Ara On', 'The Living Room',
#                         'Petterinos', 'Capriottis Sandwich Shop', 'Bibibop Asian Grill', 'Boleo', 'Pokéworks',
#                         'Primebar',
#                         'Blackwood BBQ', 'Mezcalina', 'RAISED Urban Rooftop Bar', 'Sweetwater Tavern and Grille',
#                         'Pierogi Heaven', 'The Florentine', 'New Orleans Kitchen', 'Good Stuff Eatery',
#                         'The Commons Club',
#                         'The Italian Village Restaurants', 'Stocks and Blondes', 'Ajida', 'III Forks',
#                         'Staytion Market Bar', 'Stetsons Modern Steak Sushi', 'Mortons The Steakhouse',
#                         'Giordanos Jackson',
#                         'Pizanos Pizza', 'Naansense', 'UB Dogs', 'Poke Poké', 'Rosebud Prime',
#                         'The Berghoff Restaurant',
#                         'State and Lake Chicago Tavern', 'Vapiano Chicago Downtown', 'Costa Vida Fresh Mexican Grill',
#                         'Crêpe Bistro', 'Lloyds Chicago', 'Be Leaf', 'Hannahs Bretzel', 'Ryo Sushi', 'Pueblo Chicago',
#                         'Toni Patisserie Cafe', 'Gayles Best Ever Grilled Cheese', 'Rudys Bar Grille', 'Bella Bacinos',
#                         'One North Kitchen Bar', 'Pazzos 311', 'The Walnut Room', 'City Social', 'honeygrow', 'Rivers',
#                         'The Grillroom Chophouse Wine Bar', 'Wow Bao', '1901 At AceBounce', 'Benjyehuda', 'Steadfast',
#                         'Hot Woks Cool Sushi', 'Farmers Fridge', 'sweetgreen', 'Taylor Gourmet',
#                         'Ronnys Original Chicago Steak House', 'Nutella Cafe', 'Peach And Green', 'Flat Top Grill',
#                         'IDOF I Dream of Falafel', 'Taco Bell Cantina', 'La Catina Chophouse',
#                         'Market Creations Prudential Plaza', 'Falafel Island', 'Chucks A Kerry Simon Kitchen',
#                         'Dos Toros Taqueria', 'Soup Barz', 'IPO', 'Habanero Baja Grill', 'Beef and Brandy Restaurant',
#                         'Pearl Brasserie', 'Noodles Company', 'Lockwood Restaurant Bar', 'Avanti Caffé', 'Pita Express',
#                         'The Metropolitan Club', 'Bowls Bao', 'Russian Tea Time', 'Vivere', 'Market Creations Cafe',
#                         'Monks Pub', 'Zoup', 'Just Salad', 'Chicken Planet', 'Ceres Cafe', 'Lukes Italian Beef',
#                         'Beatrix Market', 'AIRE', 'Silk Road', 'Bockwinkels', 'Bombay Wraps', 'Mercato at RustleRoux',
#                         'QDOBA Mexican Eats', 'Mezza Mediterranean Grill', 'Townhouse Restaurant Wine Bar',
#                         'Terzo Piano',
#                         'Tilted Kilt Pub Eatery', 'Park Cafe', 'The MidAmerica Club', 'Antique Taco Chiquito',
#                         'The Eastman Egg Company', 'Sweet Swabian', 'Marc Burger', 'Furious Spoon Revival Food Hall',
#                         'Emerald Loop', 'Sushi Sai', 'The Fat Shallot', 'The Budlong Revival Food Hall', 'Bob Cha',
#                         'Boston Blackies', 'Seven Bar and Restaurant', 'OBriens Riverwalk Cafe', 'I Love Sushi',
#                         'The Landing', 'Sopraffina Marketcaffe', 'Americas Dog and Burger', 'Off the Tracks Grill',
#                         'UBS Tower Cafeteria', 'Delmonico', 'The Franklin Tap', 'Market Thyme', 'La Cocina',
#                         'Taco Fresco',
#                         'Caffè Baci', 'Fontanos Subs', 'Corner Bakery Cafe', 'Halsted Street Deli', 'Bake For Me',
#                         'Cosi',
#                         'Le Pain Quotidien', 'Gold Coast Dogs', 'Metropolis', 'Ruchi', 'Olive Mediterranean Grill',
#                         'Sixty Five Chinese Restaurant', 'Cardozos Pub', 'Great Street', 'Mixed Greens', 'Als Beef',
#                         'Burrito Beach', 'Jasons Deli', 'Goddess And The Baker', 'Nesh Mediterranean Grill',
#                         'Mrs Levys Deli', 'Tokyo Lunch Boxes Catering', 'Potters', 'Potbelly Sandwich Shop',
#                         'Which Wich',
#                         'Urban Market', 'Tradition', 'Forum 55', 'Mallers Deli', 'Pazzos',
#                         'Burger Bar Chicago South Loop',
#                         'Sociale Chicago', 'Lou Malnatis Pizzeria', 'Kome Japanese Eatery', 'Armands Victory Tap',
#                         'Half Sour', 'Mercat a la Planxa', 'Sofi Restaurant', 'Bucks Four Star Grill',
#                         'South Loop Club',
#                         'Meli Cafe', 'Little Branch Cafe', 'BeeZzee Fresh Food', 'Chicago Waffles',
#                         'Plymouth Rooftop Bar Grill', 'The Scout', 'Frannies Cafe', 'Aurelios Pizza', 'Porkchop',
#                         'Mai Tai Sushi and Sake Bar', '720 South Bar Grill', 'Weather Mark Tavern',
#                         'Spanglish Mexican Kitchen', 'Hax Hops Hamburgers', 'Tutto Italiano', 'East Asian Bistro',
#                         'The Link Chicago', 'Paulys Pizzeria', 'Urban Counter', 'Bar Louie', 'The Field Bistro',
#                         'Arciel',
#                         'Eleven City Diner', 'Bulldog Ale HouseChicago', 'Poke Roll', 'The Art of Pizza',
#                         'Pats Pizzeria',
#                         'Chicago Curry House', 'Harolds Chicken Shack', 'Cafecito', 'Gioco', 'Chicago Pizza Boss',
#                         'Kim Carlos Hot Dog Stand', 'Cocos Famous Deep Fried Lobster', 'Thai Spoon and sushi', 'Niu B',
#                         'Kai Sushi', 'Umai', 'Buddy Guys Legends', 'Asian Outpost', 'Devil Dawgs South State',
#                         'Tapas Valencia', 'Villains Chicago', 'Flacos Tacos', 'Mac Kellys', 'Kitty OSheas', 'Giordanos',
#                         'Saucy Porka', 'First Draft', 'Lobo Rey', 'The Chicago Firehouse Restaurant',
#                         'Himalayan Restaurant', 'Flo Santos', 'Yolk South Loop', 'Pita Heaven',
#                         'Nepal House Indian Nepalese', 'The Bongo Room', 'Kurah Mediterranean', 'Cactus Bar Grill',
#                         'Meis Kitchen', 'Five Guys', 'Muscle Maker Grill', 'Café Press', 'Panera Bread',
#                         'Taco Burrito King', 'Relish Chicago Hot Dogs', 'Wingstop', 'Standing Room Only',
#                         'HERO Coffee Bar',
#                         'Boni Vino Ristorante Pizzeria', 'Lasalle Cafe Luna', 'DMK Burger Bar', 'Mickys Chicken Fish',
#                         'Reggies Pizza Express', 'Fountain Café', 'Greek Kitchen', 'Harrys Sandwich Shop',
#                         'La Cocina Mexican Grill', 'Field Bistro', 'Kamehachi', 'Halsted Street Deli Bagel',
#                         'Armands Pizzeria', 'Jimmy Johns', 'Girl the Goat', 'Gideon Sweet', 'Au Cheval', 'Bar Siena',
#                         'MAD Social', 'Tuscany Taylor', 'Regards to Edith', 'High Five Ramen', 'The Press Room',
#                         'Duck Duck Goat', 'Monteverde', 'Davanti Enoteca', 'The Rosebud', 'Green Street Smoked Meats',
#                         'Bad Hunter', 'Kumas Corner', 'Chilango Mexican Street Food', 'El Che Bar', 'The Publican',
#                         'Little Goat Diner', 'Cold Storage', 'Ramen Takeya', 'La Lagartija Taqueria', 'Swift Sons',
#                         'Conte Di Savoia', 'Chicken Farm Shop', 'The Office', 'Bonci', 'Francescas on Taylor', 'BLVD',
#                         'La Sirena Clandestina', 'Tufanos Vernon Park Tap', 'RoSals', 'City Mouse',
#                         'Viaggio Ristorante Lounge', 'Ricanos', 'Roister', 'Jade Court', 'Urbanbelly', 'ML Kitchen',
#                         'Leña Brava', 'The Madison Bar and Kitchen', 'Sweet Maple Cafe', 'Galata Turkish Restaurant',
#                         'Fulton Market Company', 'Guss World Famous Fried Chicken', 'GyuKaku Japanese BBQ', 'Macello',
#                         'Smyth', 'The Loyalist', 'Eggsperience', 'Federales', 'Wise Owl Drinkery Cookhouse',
#                         'RM Champagne',
#                         'Forno Rosso Pizzeria Napoletana', 'Pompei', 'Bruges Brothers', 'Next', 'Gibsons Italia',
#                         'Fulton Market Kitchen', 'Dosirak', 'Formentos', 'De Pasada', 'Busy Burger', 'Talay Chicago',
#                         'Khaosan and Taylor', 'Maudes Liquor Bar', 'bellyQ', 'Bar Takito', 'Chez Joel Bistro',
#                         'AceBounce',
#                         'Elske', 'Booze Box', 'Ciao Cafe Wine Lounge', 'Bombacignos J C Inn', 'Native Foods Cafe',
#                         'Umami Burger West Loop', 'South Branch Tavern Grille', 'Little Joes Circle Lounge',
#                         'Sushi Dokku',
#                         'Ellas Corner', 'La Sardine', 'Everest', 'Nellcôte', 'IDOF Fresh Mediterranean', 'Siam Rice',
#                         'La Taberna Tapas on Halsted', 'Parlor Pizza Bar', 'Publican Quality Meats', 'Kaiser Tiger',
#                         'Grange Hall Burger Bar', 'Lotus Cafe Bánh Mì Sandwiches', 'Yummy Thai', 'JP Graziano Grocery',
#                         'Taste222', 'The Corned Beef Factory Sandwich Shop', 'The Beer Bistro', 'Lemongrass',
#                         'Jims Original Hot Dog', 'The CrossRoads Bar Grill', 'Santorini', 'Cemitas Puebla',
#                         'Asada Mexican Grill', 'Randolph Tavern', 'Fontano Subs', 'Scafuri Bakery', 'Chinese Yum Yum',
#                         'Third Rail Tavern', 'Butterfly Sushi Bar Thai Cuisine', 'Bacci Pizzeria', 'Drum Monkey',
#                         'China Night Cafe', 'Sidebar Grille', 'Express Grill', 'Cruz Blanca Brewery Taquería', 'Nia',
#                         'Mr Broast UIC', 'Lalos Mexican Restaurant', 'Carms Beef', 'Kohan Japanese Restaurant',
#                         'Latinicity', 'West Loop Salumi', 'Hana Sushi Chinese Thai', 'Bottom Lounge', 'Wishbone',
#                         'Big Gs Pizza', 'Hello Tacos', 'Hawkeyes Bar Grill', 'Stax Cafe', 'Sushi Pink',
#                         'Frietkoten Belgian Fries Beer', 'Arturo Express', 'Cellars Market', 'Eriks Deli',
#                         'Ginos East South Loop', 'Fruve xPress Juicery', 'The Lounge', 'DoRite Donuts Chicken',
#                         'Dia De Los Tamales', 'Freshii', 'Amarit Thai Sushi', 'Morgan Street Cafe', 'ChickfilA',
#                         'Vintage Lounge', 'Palace Grill', 'Mughal India', 'M2 Cafe', 'Tiny Tapp Cafe', 'Caffe Baci',
#                         'Babas Village', 'Smoked On Taylor', 'Burrito Bravo', 'Couscous Restaurant', 'Rosatis Pizza',
#                         'Jaipur', 'Ovie Bar Grill', 'The Bar 10 Doors', 'Jackson Tavern', 'Volcano Sushi Cafe',
#                         'Alhambra Palace', 'ONeils on Wells', 'Pockets', 'Taj Mahal', 'Robinsons No 1 Ribs',
#                         'Jets Pizza',
#                         'Popeyes Louisiana Kitchen', 'Nohea Cafe', 'Dominos Pizza', 'Kimura Sushi', 'Hillshire Brands',
#                         'Au Bon Pain', 'Pittsfield Cafe', 'Tesori', 'Maxs Take Out', 'Tavern at the Park', 'Catch 35',
#                         'Mr Browns Lounge', 'Hoyts Chicago', 'McCormick Schmicks Seafood Steaks', 'The Halal Guys',
#                         'Infields', 'La Cantina', 'Simply Thalia', 'Food Court At the Miro', 'Lotus Banh Mi',
#                         '213 Cafe',
#                         'Pret a Manger', 'JJ Fish Chicken', 'Kafenio Lavazza', 'Chef Petros', 'Mr Submarine',
#                         'M Burger',
#                         'Mac Kellys Greens Things', 'Alonti Market Cafe', 'Cubano', 'Fairgrounds Coffee and Tea',
#                         'Tokyo Lunchbox', 'Piano Terra', 'Boars Head Cafe', 'Alonti Cafe', 'Falafel Express',
#                         'Mediterranean Delight', 'Milk Room', '2Twenty2 Tavern',
#                         'The Living Room Bar W Chicago City Center', 'Drawing Room', 'Red Bar', 'Happys A Holiday Bar',
#                         'Vol 39', 'M Bar', 'Brandos Speakeasy', 'Bridges Lobby Grill at London House',
#                         'Revival Cafe Bar',
#                         'Little Toasted', 'Bar Allegro', 'Kaseys Tavern', 'Two Zero Three', 'LH Rooftop', 'Game Room',
#                         'Fox Bar', 'The Bar Below', 'SkyRide Tap', 'Club Vintage',
#                         'City Winery Chicago at the Riverwalk',
#                         'Lobby Bar at Palmer House', 'MacGuffins Bar Lounge', 'Columbus Tap', 'The Junction', 'Amuse',
#                         'Prairie School', 'Waydown', 'Adamus', 'Phoenix Lounge', 'Cherry Circle Room',
#                         'Green Street Local',
#                         'Land Lake Kitchen', 'Fred Adolphs Pub', 'Lone Wolf', 'Chicago Theatre', 'The Aviary',
#                         'Blind Barber', 'Emporium Fulton Market', 'Backdoor Saloon', 'The Zodiac Room', 'The Allis',
#                         'Refuge Live', 'Chicago Union Station Metropolitan Lounge', 'Jazzin at the Shedd',
#                         'Chicago Jazz Festival', 'Tantrum', 'Pedersen Room', 'Civic Opera House', 'Congress Lounge',
#                         'Chicago Blues Festival', 'Beatrix Fulton Market', 'Auditorium Theatre', 'Artists Cafe',
#                         '231 Cafe',
#                         'Cyranos Cafe On The River', 'Lobby Lounge', 'GCue Billiards', 'Prime Matts At The Blackstone',
#                         'Rittergut Wine Bar', 'Chicago Symphony Orchestra', 'Jazz Showcase', 'Florian Opera Bistro',
#                         'Wabash Tap', 'Vice District Taproom', 'M Lounge', 'PhD Pub', 'Bracket Room', 'Square One',
#                         'Huntington Bank Pavilion', 'Taste of Randolph Street', 'Hashbrowns', 'Henry Rooftop',
#                         'The Hive',
#                         'Cettas', 'Cobra Lounge', 'WestEnd', 'The Stanley Club', 'Epiphany', 'City Winery',
#                         'CLE Cigar Company', 'The Mine Music Hall', '8fifty8', 'ThreeOneOne']

business_urls = ['/biz/bibibop-asian-grill-chicago?frvs=True&osq=Restaurants', '/biz/2twenty2-tavern-chicago?osq=Bars', '/biz/brandos-speakeasy-chicago-3?osq=Bars', '/biz/native-foods-cafe-chicago-3?frvs=True&osq=Restaurants', '/biz/plymouth-rooftop-bar-and-grill-chicago-2?osq=Bars', '/biz/ceres-cafe-chicago?frvs=True&osq=Restaurants', '/biz/exchequer-restaurant-and-pub-chicago?osq=Bars', '/biz/idof-i-dream-of-falafel-chicago-13?frvs=True&osq=Restaurants', '/biz/cellars-market-chicago?frvs=True&osq=Restaurants', '/biz/just-salad-chicago?frvs=True&osq=Restaurants', '/biz/fontanos-subs-chicago-3?osq=Takeout', '/biz/potbelly-sandwich-shop-chicago-3?frvs=True&osq=Restaurants', '/biz/harrys-sandwich-shop-chicago?frvs=True&osq=Restaurants', '/biz/als-beef-chicago-19?frvs=True&osq=Restaurants', '/biz/wow-bao-chicago-8?osq=Takeout', '/biz/muscle-maker-grill-chicago-7?frvs=True&osq=Restaurants', '/biz/the-lounge-chicago?frvs=True&osq=Restaurants', '/biz/auntie-vees-chicago-3?osq=Takeout', '/biz/king-cafe-gourmet-and-go-chicago-2?frvs=True&osq=Restaurants', '/biz/213-cafe-chicago?osq=Takeout', '/biz/market-creations-cafe-chicago-5?frvs=True&osq=Food', '/biz/jimmys-restaurant-chicago?frvs=True&osq=Restaurants', '/biz/tokyo-lunch-boxes-and-catering-chicago-7?frvs=True&osq=Restaurants', '/biz/corner-bakery-cafe-chicago-26?frvs=True&osq=Restaurants', '/biz/panda-express-chicago-13?osq=Takeout', '/biz/jimmy-johns-chicago-38?osq=Takeout', '/biz/freshii-chicago-5?osq=Takeout', '/biz/burger-king-chicago-61?frvs=True&osq=Restaurants', '/biz/qdoba-mexican-eats-chicago-7?frvs=True&osq=Restaurants', '/biz/roti-modern-mediterranean-chicago-11?frvs=True&osq=Restaurants', '/biz/chipotle-mexican-grill-chicago-31?frvs=True&osq=Restaurants', '/biz/potbelly-sandwich-shop-chicago-68?frvs=True&osq=Restaurants', '/biz/halsted-street-deli-chicago-12?frvs=True&osq=Restaurants', '/biz/potbelly-sandwich-shop-chicago-66?frvs=True&osq=Restaurants', '/biz/bacci-pizzeria-chicago-15?frvs=True&osq=Food', '/biz/potbelly-sandwich-shop-chicago-67?frvs=True&osq=Restaurants', '/biz/vivi-bubble-tea-chicago?frvs=True&osq=Food', '/biz/65-asian-kitchen-merchandise-exchange-chicago?frvs=True&osq=Food', '/biz/garrett-popcorn-shops-chicago-6?frvs=True&osq=Food', '/biz/dunkin-donuts-chicago-185?frvs=True&osq=Food', '/biz/sbarro-chicago-10?frvs=True&osq=Food', '/biz/dollop-coffee-company-chicago-2?frvs=True&osq=Food', '/biz/kilwins-chicago-2?frvs=True&osq=Food', '/biz/sopraffina-marketcaffe-chicago-6?frvs=True&osq=Food', '/biz/hero-coffee-bar-chicago-6?frvs=True&osq=Food', '/biz/mcdonalds-chicago-103?frvs=True&osq=Food', '/biz/caffe-baci-chicago-5?frvs=True&osq=Food', '/biz/eden-chicago-4?frvs=True&osq=Restaurants', '/biz/chavas-tacos-el-original-chicago?frvs=True&osq=Restaurants', '/biz/sinha-elegant-cuisine-chicago?frvs=True&osq=Restaurants', '/biz/moons-sandwich-shop-chicago?frvs=True&osq=Restaurants', '/biz/rhine-hall-distillery-chicago-2?osq=Bars', '/biz/great-central-brewing-company-chicago-2?osq=Bars', '/biz/the-ogden-chicago-4?osq=Bars', '/biz/the-slide-bar-chicago?osq=Bars', '/biz/ryhanas-cuisine-chicago?frvs=True&osq=Restaurants', '/biz/society-2201-chicago?osq=Bars', '/biz/park-tavern-chicago?osq=Bars', '/biz/jarabe-mexican-street-food-chicago?frvs=True&osq=Food', '/biz/the-corner-farmacy-chicago?frvs=True&osq=Restaurants', '/biz/king-wok-gourmet-asian-chicago?osq=Delivery', '/biz/gen-hoe-ii-chicago?osq=Delivery', '/biz/goose-island-barrel-aging-warehouse-chicago?osq=Bars', '/biz/the-grand-sandwich-stand-chicago-2?frvs=True&osq=Restaurants', '/biz/baba-pita-chicago-5?frvs=True&osq=Restaurants', '/biz/dominos-pizza-chicago-29?frvs=True&osq=Delivery', '/biz/bacci-pizzeria-chicago?frvs=True&osq=Delivery', '/biz/ashland-addison-florist-chicago-7?frvs=True&osq=Delivery', '/biz/damenzos-pizza-chicago?frvs=True&osq=Delivery', '/biz/j-and-r-express-shrimp-chicago?frvs=True&osq=Restaurants', '/biz/fiore-delicatessen-chicago?frvs=True&osq=Restaurants', '/biz/uptons-breakroom-chicago?frvs=True&osq=Restaurants', '/biz/finch-beer-co-chicago-2?osq=Bars', '/biz/jimmy-gs-restaurant-chicago?frvs=True&osq=Food', '/biz/petes-place-chicago?frvs=True&osq=Restaurants', '/biz/china-phoenix-chinese-restaurant-chicago?osq=Takeout', '/biz/als-under-the-l-chicago?frvs=True&osq=Restaurants', '/biz/mr-browns-lounge-chicago?osq=Bars', '/biz/roosters-place-chicago?osq=Bars', '/biz/wendys-chicago-42?frvs=True&osq=Restaurants', '/biz/joes-shrimp-house-chicago?frvs=True&osq=Restaurants', '/biz/budweiser-select-brew-pub-and-carvery-levy-restaurants-chicago?osq=Bars', '/biz/woodgrain-neapolitan-pizzeria-chicago?frvs=True&osq=Restaurants', '/biz/lakes-best-pizzeria-chicago?frvs=True&osq=Restaurants', '/biz/hound-dogs-burgers-and-teriyaki-chicago-2?frvs=True&osq=Restaurants', '/biz/west-side-gyros-chicago?frvs=True&osq=Restaurants', '/biz/el-original-chicago?frvs=True&osq=Restaurants', '/biz/cafe-chicago-chicago-2?frvs=True&osq=Restaurants', '/biz/jasons-wok-chicago?frvs=True&osq=Restaurants', '/biz/mcdonalds-chicago-66?frvs=True&osq=Restaurants', '/biz/ketel-one-club-chicago?frvs=True&osq=Restaurants', '/biz/harolds-chicken-shack-chicago-15?frvs=True&osq=Restaurants', '/biz/united-center-chicago?osq=Bars', '/biz/passion-house-coffee-roasters-chicago?osq=Takeout', '/biz/green-diaper-babies-chicago-2?osq=Takeout', '/biz/sharks-fish-and-chicken-chicago-15?osq=Takeout', '/biz/chicago-bone-broth-chicago?osq=Takeout', '/biz/chasing-tails-4-u-athletic-club-4-dogs-chicago?osq=Takeout', '/biz/kitchen-chicago-chicago-2?osq=Takeout', '/biz/metric-coffee-chicago?osq=Takeout', '/biz/city-view-loft-chicago?osq=Takeout', '/biz/ms-tittle-s-cupcakes-chicago?osq=Takeout', '/biz/progear-chicago?osq=Takeout', '/biz/cevapcici-chicago-chicago?osq=Takeout', '/biz/stemline-creative-chicago-2?osq=Takeout', '/biz/larkspur-chicago-2?osq=Takeout', '/biz/zen-dogs-chicago-chicago?osq=Takeout', '/biz/marc-hauser-photography-chicago-3?osq=Takeout', '/biz/northern-greenhouses-chicago?osq=Takeout', '/biz/papis-tacos-chicago-2?osq=Takeout', '/biz/paramount-events-chicago?osq=Takeout', '/biz/bc-cleaners-chicago?osq=Takeout', '/biz/j-and-j-fish-chicago?frvs=True&osq=Food', '/biz/popeyes-louisiana-kitchen-chicago-14?frvs=True&osq=Food', '/biz/wabash-seafood-company-chicago?frvs=True&osq=Food', '/biz/stock-yards-chicago?frvs=True&osq=Food', '/biz/chicago-philly-stop-chicago?frvs=True&osq=Food', '/biz/stamper-cheese-chicago?frvs=True&osq=Food', '/biz/cake-sweet-food-chicago-chicago?frvs=True&osq=Food', '/biz/taquero-fusion-chicago-3?frvs=True&osq=Food']



class YelpReviewSpider(scrapy.Spider):
    name = 'yelpreviews'

    def start_requests(self):
        baseurl = 'https://www.yelp.com'
        for url in business_urls:
            yield scrapy.Request(baseurl + url, callback=self.parse)

    def parse(self, response):
        response.selector.remove_namespaces()

        restaurant_name = response.xpath('//h1[contains(@class, "biz-page-title")]/text()').extract_first()
        restaurant_name = restaurant_name.strip()

        business_id = response.xpath('//meta[contains(@name, "yelp-biz-id")]/@content').extract_first()

        ratings = response.xpath('//div[contains(@class, "i-stars")]/@title').extract()
        reviews_ = response.xpath("//p[contains(@lang, 'en')]").extract()
        user_ids = response.xpath('//a[contains(@class, "user-display-name")]/@href').extract()
        review_ids = response.xpath('//div[contains(@class, "review")]/@data-review-id').extract()

        scores = response.xpath('//li[contains(@class, "vote-item")]/a/span/text()').extract()
        scores = list(filter(lambda x: '\n' not in x, scores))
        score_index = 0

        self.logger.info("Restaurant Name" + restaurant_name + ": Number of ratings = " + str(len(ratings)))

        for i, (rating, review_para, user_id, review_id, business_url) in enumerate(zip(ratings, reviews_, user_ids, review_ids, business_urls)):
            # clean rating
            rating = int(rating[0])

            review_para = review_para[13:-4]  # remove the <p > </p> parts
            review_para = review_para.replace('<br>', '')  # replace new lines and breaks

            # Ref: https://stackoverflow.com/questions/10993612/python-removing-xa0-from-string
            review = unicodedata.normalize('NFKD', review_para)

            user_id = user_id[21:]

            useful_key = scores[score_index]
            try:
                val = int(scores[score_index + 1])
                useful_value = val
                score_index += 2
            except:
                useful_value = 0
                score_index += 1

            funny_key = scores[score_index]
            try:
                val = int(scores[score_index + 1])
                funny_value = val
                score_index += 2
            except:
                funny_value = 0
                score_index += 1

            cool_key = scores[score_index]
            try:
                val = int(scores[score_index + 1])
                cool_value = val
                score_index += 2
            except:
                cool_value = 0
                score_index += 1


            item = {
                'restaurant_name': restaurant_name,
                'review': review,
                'rating': rating,
                'user_id': user_id,
                'business_id': business_id,
                'review_id': review_id,
                useful_key + 'Count': useful_value,
                funny_key + 'Count': funny_value,
                cool_key + 'Count': cool_value,
            }
            yield item

        current_page = response.xpath("//div[contains(@class, 'page-of-pages')]/text()").extract_first()

        if current_page is not None:
            current_page = current_page.replace('\n', '').strip().split()
            current_idx = int(current_page[1])
            total_idx = int(current_page[-1])

            if current_idx + 1 <= total_idx:
                start_count = 20 * (current_idx + 1)

                baseurl = response.url[:25]
                name = restaurant_name.split()
                name = '-'.join(name)
                name = name + '-chicago?start=%d' % start_count
                url = baseurl + name

                message = 'restaurant = ' + restaurant_name + ' | following next page to ' + url
                self.logger.info(message)

                if url is not None:
                    yield scrapy.Request(url, callback=self.parse)
