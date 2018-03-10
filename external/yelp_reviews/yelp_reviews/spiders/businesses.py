import scrapy
import unicodedata
import numpy as np

business_urls = ['/biz/bibibop-asian-grill-chicago?frvs=True&osq=Restaurants', '/biz/2twenty2-tavern-chicago?osq=Bars', '/biz/brandos-speakeasy-chicago-3?osq=Bars', '/biz/native-foods-cafe-chicago-3?frvs=True&osq=Restaurants', '/biz/plymouth-rooftop-bar-and-grill-chicago-2?osq=Bars', '/biz/ceres-cafe-chicago?frvs=True&osq=Restaurants', '/biz/exchequer-restaurant-and-pub-chicago?osq=Bars', '/biz/idof-i-dream-of-falafel-chicago-13?frvs=True&osq=Restaurants', '/biz/cellars-market-chicago?frvs=True&osq=Restaurants', '/biz/just-salad-chicago?frvs=True&osq=Restaurants', '/biz/fontanos-subs-chicago-3?osq=Takeout', '/biz/potbelly-sandwich-shop-chicago-3?frvs=True&osq=Restaurants', '/biz/harrys-sandwich-shop-chicago?frvs=True&osq=Restaurants', '/biz/als-beef-chicago-19?frvs=True&osq=Restaurants', '/biz/wow-bao-chicago-8?osq=Takeout', '/biz/muscle-maker-grill-chicago-7?frvs=True&osq=Restaurants', '/biz/the-lounge-chicago?frvs=True&osq=Restaurants', '/biz/auntie-vees-chicago-3?osq=Takeout', '/biz/king-cafe-gourmet-and-go-chicago-2?frvs=True&osq=Restaurants', '/biz/213-cafe-chicago?osq=Takeout', '/biz/market-creations-cafe-chicago-5?frvs=True&osq=Food', '/biz/jimmys-restaurant-chicago?frvs=True&osq=Restaurants', '/biz/tokyo-lunch-boxes-and-catering-chicago-7?frvs=True&osq=Restaurants', '/biz/corner-bakery-cafe-chicago-26?frvs=True&osq=Restaurants', '/biz/panda-express-chicago-13?osq=Takeout', '/biz/jimmy-johns-chicago-38?osq=Takeout', '/biz/freshii-chicago-5?osq=Takeout', '/biz/burger-king-chicago-61?frvs=True&osq=Restaurants', '/biz/qdoba-mexican-eats-chicago-7?frvs=True&osq=Restaurants', '/biz/roti-modern-mediterranean-chicago-11?frvs=True&osq=Restaurants', '/biz/chipotle-mexican-grill-chicago-31?frvs=True&osq=Restaurants', '/biz/potbelly-sandwich-shop-chicago-68?frvs=True&osq=Restaurants', '/biz/halsted-street-deli-chicago-12?frvs=True&osq=Restaurants', '/biz/potbelly-sandwich-shop-chicago-66?frvs=True&osq=Restaurants', '/biz/bacci-pizzeria-chicago-15?frvs=True&osq=Food', '/biz/potbelly-sandwich-shop-chicago-67?frvs=True&osq=Restaurants', '/biz/vivi-bubble-tea-chicago?frvs=True&osq=Food', '/biz/65-asian-kitchen-merchandise-exchange-chicago?frvs=True&osq=Food', '/biz/garrett-popcorn-shops-chicago-6?frvs=True&osq=Food', '/biz/dunkin-donuts-chicago-185?frvs=True&osq=Food', '/biz/sbarro-chicago-10?frvs=True&osq=Food', '/biz/dollop-coffee-company-chicago-2?frvs=True&osq=Food', '/biz/kilwins-chicago-2?frvs=True&osq=Food', '/biz/sopraffina-marketcaffe-chicago-6?frvs=True&osq=Food', '/biz/hero-coffee-bar-chicago-6?frvs=True&osq=Food', '/biz/mcdonalds-chicago-103?frvs=True&osq=Food', '/biz/caffe-baci-chicago-5?frvs=True&osq=Food', '/biz/eden-chicago-4?frvs=True&osq=Restaurants', '/biz/chavas-tacos-el-original-chicago?frvs=True&osq=Restaurants', '/biz/sinha-elegant-cuisine-chicago?frvs=True&osq=Restaurants', '/biz/moons-sandwich-shop-chicago?frvs=True&osq=Restaurants', '/biz/rhine-hall-distillery-chicago-2?osq=Bars', '/biz/great-central-brewing-company-chicago-2?osq=Bars', '/biz/the-ogden-chicago-4?osq=Bars', '/biz/the-slide-bar-chicago?osq=Bars', '/biz/ryhanas-cuisine-chicago?frvs=True&osq=Restaurants', '/biz/society-2201-chicago?osq=Bars', '/biz/park-tavern-chicago?osq=Bars', '/biz/jarabe-mexican-street-food-chicago?frvs=True&osq=Food', '/biz/the-corner-farmacy-chicago?frvs=True&osq=Restaurants', '/biz/king-wok-gourmet-asian-chicago?osq=Delivery', '/biz/gen-hoe-ii-chicago?osq=Delivery', '/biz/goose-island-barrel-aging-warehouse-chicago?osq=Bars', '/biz/the-grand-sandwich-stand-chicago-2?frvs=True&osq=Restaurants', '/biz/baba-pita-chicago-5?frvs=True&osq=Restaurants', '/biz/dominos-pizza-chicago-29?frvs=True&osq=Delivery', '/biz/bacci-pizzeria-chicago?frvs=True&osq=Delivery', '/biz/ashland-addison-florist-chicago-7?frvs=True&osq=Delivery', '/biz/damenzos-pizza-chicago?frvs=True&osq=Delivery', '/biz/j-and-r-express-shrimp-chicago?frvs=True&osq=Restaurants', '/biz/fiore-delicatessen-chicago?frvs=True&osq=Restaurants', '/biz/uptons-breakroom-chicago?frvs=True&osq=Restaurants', '/biz/finch-beer-co-chicago-2?osq=Bars', '/biz/jimmy-gs-restaurant-chicago?frvs=True&osq=Food', '/biz/petes-place-chicago?frvs=True&osq=Restaurants', '/biz/china-phoenix-chinese-restaurant-chicago?osq=Takeout', '/biz/als-under-the-l-chicago?frvs=True&osq=Restaurants', '/biz/mr-browns-lounge-chicago?osq=Bars', '/biz/roosters-place-chicago?osq=Bars', '/biz/wendys-chicago-42?frvs=True&osq=Restaurants', '/biz/joes-shrimp-house-chicago?frvs=True&osq=Restaurants', '/biz/budweiser-select-brew-pub-and-carvery-levy-restaurants-chicago?osq=Bars', '/biz/woodgrain-neapolitan-pizzeria-chicago?frvs=True&osq=Restaurants', '/biz/lakes-best-pizzeria-chicago?frvs=True&osq=Restaurants', '/biz/hound-dogs-burgers-and-teriyaki-chicago-2?frvs=True&osq=Restaurants', '/biz/west-side-gyros-chicago?frvs=True&osq=Restaurants', '/biz/el-original-chicago?frvs=True&osq=Restaurants', '/biz/cafe-chicago-chicago-2?frvs=True&osq=Restaurants', '/biz/jasons-wok-chicago?frvs=True&osq=Restaurants', '/biz/mcdonalds-chicago-66?frvs=True&osq=Restaurants', '/biz/ketel-one-club-chicago?frvs=True&osq=Restaurants', '/biz/harolds-chicken-shack-chicago-15?frvs=True&osq=Restaurants', '/biz/united-center-chicago?osq=Bars', '/biz/passion-house-coffee-roasters-chicago?osq=Takeout', '/biz/green-diaper-babies-chicago-2?osq=Takeout', '/biz/sharks-fish-and-chicken-chicago-15?osq=Takeout', '/biz/chicago-bone-broth-chicago?osq=Takeout', '/biz/chasing-tails-4-u-athletic-club-4-dogs-chicago?osq=Takeout', '/biz/kitchen-chicago-chicago-2?osq=Takeout', '/biz/metric-coffee-chicago?osq=Takeout', '/biz/city-view-loft-chicago?osq=Takeout', '/biz/ms-tittle-s-cupcakes-chicago?osq=Takeout', '/biz/progear-chicago?osq=Takeout', '/biz/cevapcici-chicago-chicago?osq=Takeout', '/biz/stemline-creative-chicago-2?osq=Takeout', '/biz/larkspur-chicago-2?osq=Takeout', '/biz/zen-dogs-chicago-chicago?osq=Takeout', '/biz/marc-hauser-photography-chicago-3?osq=Takeout', '/biz/northern-greenhouses-chicago?osq=Takeout', '/biz/papis-tacos-chicago-2?osq=Takeout', '/biz/paramount-events-chicago?osq=Takeout', '/biz/bc-cleaners-chicago?osq=Takeout', '/biz/j-and-j-fish-chicago?frvs=True&osq=Food', '/biz/popeyes-louisiana-kitchen-chicago-14?frvs=True&osq=Food', '/biz/wabash-seafood-company-chicago?frvs=True&osq=Food', '/biz/stock-yards-chicago?frvs=True&osq=Food', '/biz/chicago-philly-stop-chicago?frvs=True&osq=Food', '/biz/stamper-cheese-chicago?frvs=True&osq=Food', '/biz/cake-sweet-food-chicago-chicago?frvs=True&osq=Food', '/biz/taquero-fusion-chicago-3?frvs=True&osq=Food']

def clean_strings(x):
    x = x.replace("\n", "")
    x = x.strip()
    return x


def convert_to_bool(x):
    x = str(x).lower()
    if x in 'no':
        return False
    else:
        return True


class YelpBusinessSpider(scrapy.Spider):
    name = 'yelp_business'

    def start_requests(self):
        baseurl = 'https://www.yelp.com'
        for url in business_urls:
            yield scrapy.Request(baseurl + url, callback=self.parse)

    def parse(self, response):
        restaurant = response.url[25:].split('-')
        restaurant_name = ' '.join(restaurant[:-1])

        response.selector.remove_namespaces()

        business_id = response.xpath('//meta[contains(@name, "yelp-biz-id")]/@content').extract_first()

        self.logger.info("Restaurant Name" + restaurant_name)

        dates_span = response.xpath('//span[contains(@class, "nowrap")]/text()').extract()

        dates = []
        for i in range(len(dates_span)):
            if 'am' in dates_span[i] or 'pm' in dates_span[i]:
                dates.append(dates_span[i])

        dates = dates[:14]

        hours = []
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for i in range(0, len(dates), 2):
            start = dates[i]
            end = dates[i + 1]
            day = days[i // 2]
            string = day + ':' + start + '-' + end
            hours.append(string)

        hours = ','.join(hours)

        website = response.xpath('//a[contains(@href, "/biz_redir?")]/text()').extract_first()
        if website is None:
            website = None

        phone_number = response.xpath('//span[contains(@class, "biz-phone")]/text()').extract_first()
        if phone_number is not None:
            phone_number = phone_number.replace("\n", "").strip()

        attribute_keys = response.xpath('//dt[contains(@class, "attribute-key")]/text()').extract()
        attribute_keys = list(map(clean_strings, filter(lambda x: "\n" in x, attribute_keys)))

        attribute_values = response.xpath('//dd/text()').extract()
        attribute_values = attribute_values[-len(attribute_keys):]

        attribute_keys = list(map(clean_strings, attribute_keys))
        attribute_values = list(map(clean_strings, attribute_values))
        attribute_values = list(map(convert_to_bool, attribute_values))

        attribute_map = {}
        for i, (key, value) in enumerate(zip(attribute_keys, attribute_values)):
            attribute_map[key] = value

        store_dict = {
            'restaurantID': business_id,
            'hours': hours,
            'website': website,
            'phoneNumber': phone_number,
        }
        store_dict.update(attribute_map)

        yield store_dict