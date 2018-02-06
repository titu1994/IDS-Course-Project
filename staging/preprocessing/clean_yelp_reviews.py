import pandas as pd
import numpy as np
import re

from staging.utils.generic_utils import resolve_data_path, construct_data_path

restaurant_names = ['The Dearborn', 'Stock and Ledger', 'Remingtons', 'Roanoke Restaurant', 'The Gage', 'The Marq',
                    'MingHin Cuisine', 'Nandos PERiPERi', 'Cochon Volant Brasserie', 'Green Apple PHOever',
                    'Ge Pa De Caffe', 'Roti Modern Mediterranean', 'Shake Shack', 'Pastoral', 'Seven Lions',
                    'Wildberry Pancakes and Cafe', 'Heaven on Seven Wabash', 'Millers Pub',
                    'American Craft Kitchen Bar', 'Oasis Cafe', 'Filini', '312 Chicago', 'Brown Bag Seafood',
                    'Union Squared', 'Osaka Sushi Express Fresh Fruit Smoothies', 'Billy Goat Tavern Lake Street',
                    'Eggys Diner', 'Exchequer Restaurant Pub', 'Protein Bar', 'Tokyo Lunch Boxes', 'Atwood',
                    'Elephant Castle', 'La Bamba', 'Park Grill at Millennium Park', 'Early Society',
                    'University Club of Chicago', 'BenjYehuda', 'Spotted Monkey', 'Frontera Fresco', 'Prime Provisions',
                    'Lukes Lobster City Hall', 'Acanto', 'Brightwok Kitchen', 'Trattoria No 10', 'Naf Naf Grill',
                    'Cindys', 'Broken English Taco Pub The Loop', 'Revival Food Hall', 'Ara On', 'The Living Room',
                    'Petterinos', 'Capriottis Sandwich Shop', 'Bibibop Asian Grill', 'Boleo', 'Pokéworks', 'Primebar',
                    'Blackwood BBQ', 'Mezcalina', 'RAISED Urban Rooftop Bar', 'Sweetwater Tavern and Grille',
                    'Pierogi Heaven', 'The Florentine', 'New Orleans Kitchen', 'Good Stuff Eatery', 'The Commons Club',
                    'The Italian Village Restaurants', 'Stocks and Blondes', 'Ajida', 'III Forks',
                    'Staytion Market Bar', 'Stetsons Modern Steak Sushi', 'Mortons The Steakhouse', 'Giordanos Jackson',
                    'Pizanos Pizza', 'Naansense', 'UB Dogs', 'Poke Poké', 'Rosebud Prime', 'The Berghoff Restaurant',
                    'State and Lake Chicago Tavern', 'Vapiano Chicago Downtown', 'Costa Vida Fresh Mexican Grill',
                    'Crêpe Bistro', 'Lloyds Chicago', 'Be Leaf', 'Hannahs Bretzel', 'Ryo Sushi', 'Pueblo Chicago',
                    'Toni Patisserie Cafe', 'Gayles Best Ever Grilled Cheese', 'Rudys Bar Grille', 'Bella Bacinos',
                    'One North Kitchen Bar', 'Pazzos 311', 'The Walnut Room', 'City Social', 'honeygrow', 'Rivers',
                    'The Grillroom Chophouse Wine Bar', 'Wow Bao', '1901 At AceBounce', 'Benjyehuda', 'Steadfast',
                    'Hot Woks Cool Sushi', 'Farmers Fridge', 'sweetgreen', 'Taylor Gourmet',
                    'Ronnys Original Chicago Steak House', 'Nutella Cafe', 'Peach And Green', 'Flat Top Grill',
                    'IDOF I Dream of Falafel', 'Taco Bell Cantina', 'La Catina Chophouse',
                    'Market Creations Prudential Plaza', 'Falafel Island', 'Chucks A Kerry Simon Kitchen',
                    'Dos Toros Taqueria', 'Soup Barz', 'IPO', 'Habanero Baja Grill', 'Beef and Brandy Restaurant',
                    'Pearl Brasserie', 'Noodles Company', 'Lockwood Restaurant Bar', 'Avanti Caffé', 'Pita Express',
                    'The Metropolitan Club', 'Bowls Bao', 'Russian Tea Time', 'Vivere', 'Market Creations Cafe',
                    'Monks Pub', 'Zoup', 'Just Salad', 'Chicken Planet', 'Ceres Cafe', 'Lukes Italian Beef',
                    'Beatrix Market', 'AIRE', 'Silk Road', 'Bockwinkels', 'Bombay Wraps', 'Mercato at RustleRoux',
                    'QDOBA Mexican Eats', 'Mezza Mediterranean Grill', 'Townhouse Restaurant Wine Bar', 'Terzo Piano',
                    'Tilted Kilt Pub Eatery', 'Park Cafe', 'The MidAmerica Club', 'Antique Taco Chiquito',
                    'The Eastman Egg Company', 'Sweet Swabian', 'Marc Burger', 'Furious Spoon Revival Food Hall',
                    'Emerald Loop', 'Sushi Sai', 'The Fat Shallot', 'The Budlong Revival Food Hall', 'Bob Cha',
                    'Boston Blackies', 'Seven Bar and Restaurant', 'OBriens Riverwalk Cafe', 'I Love Sushi',
                    'The Landing', 'Sopraffina Marketcaffe', 'Americas Dog and Burger', 'Off the Tracks Grill',
                    'UBS Tower Cafeteria', 'Delmonico', 'The Franklin Tap', 'Market Thyme', 'La Cocina', 'Taco Fresco',
                    'Caffè Baci', 'Fontanos Subs', 'Corner Bakery Cafe', 'Halsted Street Deli', 'Bake For Me', 'Cosi',
                    'Le Pain Quotidien', 'Gold Coast Dogs', 'Metropolis', 'Ruchi', 'Olive Mediterranean Grill',
                    'Sixty Five Chinese Restaurant', 'Cardozos Pub', 'Great Street', 'Mixed Greens', 'Als Beef',
                    'Burrito Beach', 'Jasons Deli', 'Goddess And The Baker', 'Nesh Mediterranean Grill',
                    'Mrs Levys Deli', 'Tokyo Lunch Boxes Catering', 'Potters', 'Potbelly Sandwich Shop', 'Which Wich',
                    'Urban Market', 'Tradition', 'Forum 55', 'Mallers Deli', 'Pazzos', 'Burger Bar Chicago South Loop',
                    'Sociale Chicago', 'Lou Malnatis Pizzeria', 'Kome Japanese Eatery', 'Armands Victory Tap',
                    'Half Sour', 'Mercat a la Planxa', 'Sofi Restaurant', 'Bucks Four Star Grill', 'South Loop Club',
                    'Meli Cafe', 'Little Branch Cafe', 'BeeZzee Fresh Food', 'Chicago Waffles',
                    'Plymouth Rooftop Bar Grill', 'The Scout', 'Frannies Cafe', 'Aurelios Pizza', 'Porkchop',
                    'Mai Tai Sushi and Sake Bar', '720 South Bar Grill', 'Weather Mark Tavern',
                    'Spanglish Mexican Kitchen', 'Hax Hops Hamburgers', 'Tutto Italiano', 'East Asian Bistro',
                    'The Link Chicago', 'Paulys Pizzeria', 'Urban Counter', 'Bar Louie', 'The Field Bistro', 'Arciel',
                    'Eleven City Diner', 'Bulldog Ale HouseChicago', 'Poke Roll', 'The Art of Pizza', 'Pats Pizzeria',
                    'Chicago Curry House', 'Harolds Chicken Shack', 'Cafecito', 'Gioco', 'Chicago Pizza Boss',
                    'Kim Carlos Hot Dog Stand', 'Cocos Famous Deep Fried Lobster', 'Thai Spoon and sushi', 'Niu B',
                    'Kai Sushi', 'Umai', 'Buddy Guys Legends', 'Asian Outpost', 'Devil Dawgs South State',
                    'Tapas Valencia', 'Villains Chicago', 'Flacos Tacos', 'Mac Kellys', 'Kitty OSheas', 'Giordanos',
                    'Saucy Porka', 'First Draft', 'Lobo Rey', 'The Chicago Firehouse Restaurant',
                    'Himalayan Restaurant', 'Flo Santos', 'Yolk South Loop', 'Pita Heaven',
                    'Nepal House Indian Nepalese', 'The Bongo Room', 'Kurah Mediterranean', 'Cactus Bar Grill',
                    'Meis Kitchen', 'Five Guys', 'Muscle Maker Grill', 'Café Press', 'Panera Bread',
                    'Taco Burrito King', 'Relish Chicago Hot Dogs', 'Wingstop', 'Standing Room Only', 'HERO Coffee Bar',
                    'Boni Vino Ristorante Pizzeria', 'Lasalle Cafe Luna', 'DMK Burger Bar', 'Mickys Chicken Fish',
                    'Reggies Pizza Express', 'Fountain Café', 'Greek Kitchen', 'Harrys Sandwich Shop',
                    'La Cocina Mexican Grill', 'Field Bistro', 'Kamehachi', 'Halsted Street Deli Bagel',
                    'Armands Pizzeria', 'Jimmy Johns', 'Girl the Goat', 'Gideon Sweet', 'Au Cheval', 'Bar Siena',
                    'MAD Social', 'Tuscany Taylor', 'Regards to Edith', 'High Five Ramen', 'The Press Room',
                    'Duck Duck Goat', 'Monteverde', 'Davanti Enoteca', 'The Rosebud', 'Green Street Smoked Meats',
                    'Bad Hunter', 'Kumas Corner', 'Chilango Mexican Street Food', 'El Che Bar', 'The Publican',
                    'Little Goat Diner', 'Cold Storage', 'Ramen Takeya', 'La Lagartija Taqueria', 'Swift Sons',
                    'Conte Di Savoia', 'Chicken Farm Shop', 'The Office', 'Bonci', 'Francescas on Taylor', 'BLVD',
                    'La Sirena Clandestina', 'Tufanos Vernon Park Tap', 'RoSals', 'City Mouse',
                    'Viaggio Ristorante Lounge', 'Ricanos', 'Roister', 'Jade Court', 'Urbanbelly', 'ML Kitchen',
                    'Leña Brava', 'The Madison Bar and Kitchen', 'Sweet Maple Cafe', 'Galata Turkish Restaurant',
                    'Fulton Market Company', 'Guss World Famous Fried Chicken', 'GyuKaku Japanese BBQ', 'Macello',
                    'Smyth', 'The Loyalist', 'Eggsperience', 'Federales', 'Wise Owl Drinkery Cookhouse', 'RM Champagne',
                    'Forno Rosso Pizzeria Napoletana', 'Pompei', 'Bruges Brothers', 'Next', 'Gibsons Italia',
                    'Fulton Market Kitchen', 'Dosirak', 'Formentos', 'De Pasada', 'Busy Burger', 'Talay Chicago',
                    'Khaosan and Taylor', 'Maudes Liquor Bar', 'bellyQ', 'Bar Takito', 'Chez Joel Bistro', 'AceBounce',
                    'Elske', 'Booze Box', 'Ciao Cafe Wine Lounge', 'Bombacignos J C Inn', 'Native Foods Cafe',
                    'Umami Burger West Loop', 'South Branch Tavern Grille', 'Little Joes Circle Lounge', 'Sushi Dokku',
                    'Ellas Corner', 'La Sardine', 'Everest', 'Nellcôte', 'IDOF Fresh Mediterranean', 'Siam Rice',
                    'La Taberna Tapas on Halsted', 'Parlor Pizza Bar', 'Publican Quality Meats', 'Kaiser Tiger',
                    'Grange Hall Burger Bar', 'Lotus Cafe Bánh Mì Sandwiches', 'Yummy Thai', 'JP Graziano Grocery',
                    'Taste222', 'The Corned Beef Factory Sandwich Shop', 'The Beer Bistro', 'Lemongrass',
                    'Jims Original Hot Dog', 'The CrossRoads Bar Grill', 'Santorini', 'Cemitas Puebla',
                    'Asada Mexican Grill', 'Randolph Tavern', 'Fontano Subs', 'Scafuri Bakery', 'Chinese Yum Yum',
                    'Third Rail Tavern', 'Butterfly Sushi Bar Thai Cuisine', 'Bacci Pizzeria', 'Drum Monkey',
                    'China Night Cafe', 'Sidebar Grille', 'Express Grill', 'Cruz Blanca Brewery Taquería', 'Nia',
                    'Mr Broast UIC', 'Lalos Mexican Restaurant', 'Carms Beef', 'Kohan Japanese Restaurant',
                    'Latinicity', 'West Loop Salumi', 'Hana Sushi Chinese Thai', 'Bottom Lounge', 'Wishbone',
                    'Big Gs Pizza', 'Hello Tacos', 'Hawkeyes Bar Grill', 'Stax Cafe', 'Sushi Pink',
                    'Frietkoten Belgian Fries Beer', 'Arturo Express', 'Cellars Market', 'Eriks Deli',
                    'Ginos East South Loop', 'Fruve xPress Juicery', 'The Lounge', 'DoRite Donuts Chicken',
                    'Dia De Los Tamales', 'Freshii', 'Amarit Thai Sushi', 'Morgan Street Cafe', 'ChickfilA',
                    'Vintage Lounge', 'Palace Grill', 'Mughal India', 'M2 Cafe', 'Tiny Tapp Cafe', 'Caffe Baci',
                    'Babas Village', 'Smoked On Taylor', 'Burrito Bravo', 'Couscous Restaurant', 'Rosatis Pizza',
                    'Jaipur', 'Ovie Bar Grill', 'The Bar 10 Doors', 'Jackson Tavern', 'Volcano Sushi Cafe',
                    'Alhambra Palace', 'ONeils on Wells', 'Pockets', 'Taj Mahal', 'Robinsons No 1 Ribs', 'Jets Pizza',
                    'Popeyes Louisiana Kitchen', 'Nohea Cafe', 'Dominos Pizza', 'Kimura Sushi', 'Hillshire Brands',
                    'Au Bon Pain', 'Pittsfield Cafe', 'Tesori', 'Maxs Take Out', 'Tavern at the Park', 'Catch 35',
                    'Mr Browns Lounge', 'Hoyts Chicago', 'McCormick Schmicks Seafood Steaks', 'The Halal Guys',
                    'Infields', 'La Cantina', 'Simply Thalia', 'Food Court At the Miro', 'Lotus Banh Mi', '213 Cafe',
                    'Pret a Manger', 'JJ Fish Chicken', 'Kafenio Lavazza', 'Chef Petros', 'Mr Submarine', 'M Burger',
                    'Mac Kellys Greens Things', 'Alonti Market Cafe', 'Cubano', 'Fairgrounds Coffee and Tea',
                    'Tokyo Lunchbox', 'Piano Terra', 'Boars Head Cafe', 'Alonti Cafe', 'Falafel Express',
                    'Mediterranean Delight', 'Milk Room', '2Twenty2 Tavern',
                    'The Living Room Bar W Chicago City Center', 'Drawing Room', 'Red Bar', 'Happys A Holiday Bar',
                    'Vol 39', 'M Bar', 'Brandos Speakeasy', 'Bridges Lobby Grill at London House', 'Revival Cafe Bar',
                    'Little Toasted', 'Bar Allegro', 'Kaseys Tavern', 'Two Zero Three', 'LH Rooftop', 'Game Room',
                    'Fox Bar', 'The Bar Below', 'SkyRide Tap', 'Club Vintage', 'City Winery Chicago at the Riverwalk',
                    'Lobby Bar at Palmer House', 'MacGuffins Bar Lounge', 'Columbus Tap', 'The Junction', 'Amuse',
                    'Prairie School', 'Waydown', 'Adamus', 'Phoenix Lounge', 'Cherry Circle Room', 'Green Street Local',
                    'Land Lake Kitchen', 'Fred Adolphs Pub', 'Lone Wolf', 'Chicago Theatre', 'The Aviary',
                    'Blind Barber', 'Emporium Fulton Market', 'Backdoor Saloon', 'The Zodiac Room', 'The Allis',
                    'Refuge Live', 'Chicago Union Station Metropolitan Lounge', 'Jazzin at the Shedd',
                    'Chicago Jazz Festival', 'Tantrum', 'Pedersen Room', 'Civic Opera House', 'Congress Lounge',
                    'Chicago Blues Festival', 'Beatrix Fulton Market', 'Auditorium Theatre', 'Artists Cafe', '231 Cafe',
                    'Cyranos Cafe On The River', 'Lobby Lounge', 'GCue Billiards', 'Prime Matts At The Blackstone',
                    'Rittergut Wine Bar', 'Chicago Symphony Orchestra', 'Jazz Showcase', 'Florian Opera Bistro',
                    'Wabash Tap', 'Vice District Taproom', 'M Lounge', 'PhD Pub', 'Bracket Room', 'Square One',
                    'Huntington Bank Pavilion', 'Taste of Randolph Street', 'Hashbrowns', 'Henry Rooftop', 'The Hive',
                    'Cettas', 'Cobra Lounge', 'WestEnd', 'The Stanley Club', 'Epiphany', 'City Winery',
                    'CLE Cigar Company', 'The Mine Music Hall', '8fifty8', 'ThreeOneOne']

restaurant_names_temp = []
for name in restaurant_names:
    name = name.lower()
    restaurant_names_temp.append(name)
restaurant_names = restaurant_names_temp

basepath = "raw/yelp-reviews/yelp_reviews.json"
path = resolve_data_path(basepath)

df = pd.read_json(path)  # type: pd.DataFrame

names = df.restaurant_name.values
names = np.unique(names)

print('Number of unique restaurants with reviews:', len(names), '\n')
loaded_restaurants = set(names.tolist())
all_restaurants = set(restaurant_names)

restaurants_remaining = all_restaurants - loaded_restaurants

print()
print('number of restaurants remaining:', len(list(restaurants_remaining)))
print(list(restaurants_remaining))