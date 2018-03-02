import scrapy
import unicodedata
import numpy as np

business_urls = ['/biz/bibibop-asian-grill-chicago?frvs=True&osq=Restaurants', '/biz/native-foods-cafe-chicago-3?frvs=True&osq=Restaurants', '/biz/ceres-cafe-chicago?frvs=True&osq=Restaurants', '/biz/plymouth-rooftop-bar-and-grill-chicago-2?frvs=True&osq=Restaurants', '/biz/idof-i-dream-of-falafel-chicago-13?frvs=True&osq=Restaurants', '/biz/cellars-market-chicago?frvs=True&osq=Restaurants', '/biz/just-salad-chicago?frvs=True&osq=Restaurants', '/biz/exchequer-restaurant-and-pub-chicago?frvs=True&osq=Restaurants', '/biz/fontanos-subs-chicago-3?frvs=True&osq=Restaurants', '/biz/potbelly-sandwich-shop-chicago-3?frvs=True&osq=Restaurants', '/biz/harrys-sandwich-shop-chicago?frvs=True&osq=Restaurants', '/biz/als-beef-chicago-19?frvs=True&osq=Restaurants', '/biz/muscle-maker-grill-chicago-7?frvs=True&osq=Restaurants', '/biz/wow-bao-chicago-8?frvs=True&osq=Restaurants', '/biz/the-lounge-chicago?frvs=True&osq=Restaurants', '/biz/king-cafe-gourmet-and-go-chicago-2?frvs=True&osq=Restaurants', '/biz/jimmys-restaurant-chicago?frvs=True&osq=Restaurants', '/biz/tokyo-lunch-boxes-and-catering-chicago-7?frvs=True&osq=Restaurants', '/biz/corner-bakery-cafe-chicago-26?frvs=True&osq=Restaurants', '/biz/213-cafe-chicago?frvs=True&osq=Restaurants', '/biz/burger-king-chicago-61?frvs=True&osq=Restaurants', '/biz/qdoba-mexican-eats-chicago-7?frvs=True&osq=Restaurants', '/biz/roti-modern-mediterranean-chicago-11?frvs=True&osq=Restaurants', '/biz/market-creations-cafe-chicago-5?frvs=True&osq=Restaurants', '/biz/chipotle-mexican-grill-chicago-31?frvs=True&osq=Restaurants', '/biz/halsted-street-deli-chicago-12?frvs=True&osq=Restaurants', '/biz/jimmy-johns-chicago-38?frvs=True&osq=Restaurants', '/biz/freshii-chicago-5?frvs=True&osq=Restaurants']


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