import scrapy
import unicodedata

locations = ["Chicago,+IL+{}", "{}"]
pincodes = [60612, 60604]
types = ['Restaurants', 'Bars', 'Food', 'Delivery', 'Takeout', 'Reservations']
#
# class YelpDataSpider(scrapy.Spider):
#     name = 'yelp_data_old'
#
#     def start_requests(self):
#         baseurl = 'https://www.yelp.com/search?'
#
#         for search in types:
#             for location in locations:
#                 for pincode in pincodes:
#                     location = location % (pincode)
#                     url = baseurl + 'find_desc=' + search + '&find_loc=' + location
#                     yield scrapy.Request(url, callback=self.parse)
#
#     def parse(self, response):
#         for place in response.css("li.regular-search-result"):
#             yield {
#                 'Name': unicodedata.normalize('NFKD', place.xpath("div/div[1]/div[1]/div/div[2]/h3/span/a/span/text()").extract_first()).strip(),
#                 'Ratings': place.xpath("div/div[1]/div[1]/div/div[2]/div[1]/div/@title").extract_first()
#                     .replace(" star rating", ""),
#                 'Reviews': place.xpath("div/div[1]/div[1]/div/div[2]/div[1]/span/text()").extract_first()
#                     .strip().replace(" reviews", ""),
#                 'Price': place.xpath("div/div[1]/div[1]/div/div[2]/div[2]/span[1]/span/text()").extract(),
#                 'Category': place.xpath("div/div[1]/div[1]/div/div[2]/div[2]/span[2]/a/text()").extract(),
#                 'Neighbourhood': place.xpath("div/div[1]/div[2]/span[1]/text()").extract_first().strip(),
#                 'Address': ", ".join(place.xpath("div/div[1]/div[2]/address/text()").extract()).strip(),
#                 'Phone': place.xpath("div/div[1]/div[2]/span[3]/text()").extract_first().strip(),
#             }
#
#         for next in response.css("a.pagination-links_anchor"):
#             yield response.follow(next, callback=self.parse)


class YelpDataSpider2(scrapy.Spider):
    name = 'yelp_data'

    def start_requests(self):
        baseurl = 'https://www.yelp.com/search?'

        for search in types:
            for location in locations:
                for pincode in pincodes:
                    location = location.format(pincode)
                    url = baseurl + 'find_desc=' + search + '&find_loc=' + location + "&start=0"

                    self.logger.info("\n\nURL Searching : " + url + "\n\n")

                    yield scrapy.Request(url, callback=self.parse)

    def parse(self, response):
        for place in response.css("li.regular-search-result"):
            try:
                name = unicodedata.normalize('NFKD', place.xpath("div/div[1]/div[1]/div/div[2]/h3/span/a/span/text()").extract_first()).strip()
            except:
                continue

            ratings = place.xpath("div/div[1]/div[1]/div/div[2]/div[1]/div/@title").extract_first().replace(" star rating", "")

            reviews = place.xpath("div/div[1]/div[1]/div/div[2]/div[1]/span/text()").extract_first().strip().replace(" reviews", "")

            price = place.xpath("div/div[1]/div[1]/div/div[2]/div[2]/span[1]/span/text()").extract()

            category = place.xpath("div/div[1]/div[1]/div/div[2]/div[2]/span[2]/a/text()").extract()

            neighbourhood = place.xpath("div/div[1]/div[2]/span[1]/text()").extract_first().strip()

            address = ", ".join(place.xpath("div/div[1]/div[2]/address/text()").extract()).strip()

            business_url = place.xpath('div/div[1]/div[1]/div/div[2]/h3/span/a/@href').extract_first()

            business_id = place.xpath('div/@data-biz-id').extract_first()

            phone = place.xpath("div/div[1]/div[2]/span[3]/text()").extract_first()
            if phone is not None:
                phone = phone.strip()

            yield {
                'Name': name,
                'Ratings': ratings,
                'ReviewCount': reviews,
                'Price': price,
                'Category': category,
                'Neighbourhood': neighbourhood,
                'Address': address,
                'Phone': phone,
                'YelpURL': business_url,
                'BusinessID': business_id,
            }

        response.selector.remove_namespaces()

        current_page = response.xpath("//div[contains(@class, 'page-of-pages')]/text()").extract_first()

        self.logger.info("Current page : " + current_page)

        if current_page is not None:
            current_page = current_page.replace('\n', '').strip().split()
            current_idx = int(current_page[1])
            total_idx = int(current_page[-1])

            if current_idx <= total_idx:
                prev_count = 10 * (current_idx - 1)
                start_count = 10 * (current_idx)

                baseurl = response.url.replace("&start={}".format(prev_count), "&start={}".format(start_count))

                self.logger.info("BaseURL : " + baseurl)

                yield scrapy.Request(baseurl, callback=self.parse)