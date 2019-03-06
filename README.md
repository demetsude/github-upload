# CTR Prediction based on the Data of Ad Campaigns Performances

A simple regression model to predict the CTR for advertisements

### Predictor Class
* The model is implemented as a Python class. The methods are explained as follows:
	* Remove missing data
		* There are many missing values in the last two columns, `total_conversion` and `approved_conversion`. But if we carefully look at the rows with the missing values, it is understood that the missing values belong to different columns, `campaign_id` and `fb_campaign_id`. 
		* Since `fb_campaign_id` column contains unique ids, it is not included as a feature. Additionally, `campaign_id` is not considered since it is not a configurable parameter of a campaign.
		* The related columns with the missing values are deleted and the whole file is rearranged.

	* Prepare data
		* Independent Variables
			* Categorical Variables: 
				* Week Day: The day of the week might be useful in terms of giving insight about the users' click behavior based on different days of the week. Therefore, the day of the week is extracted from the column `reporting_start`. 
				* Gender: The `gender` variable is a categorical data and given as a string. It is encoded and used as a feature. 
				* Interest1: The column `interest1`, `interest2` and `interest3` are given as a type of integer. But these numbers do not have an inherent ordinality based on their numerical values. In other words, these are categorical variables. Therefore, they are converted to a string. Among them, the column `interest1` is used only. If we consider them as integer values, it is obvious that there is a strong correlation between them. It also applies for them as categorical variables. They together decrease the quality of predictions as they contain the same information. After investigating the effect of them, the best column is chosen as `interest1`.
			* Numerical Variables: 
				* Day: The columns `reporting_start` and `reporting_end` contain the same information and all the dates are in the same month and year. Therefore, the only useful information is day and it is extracted and added as a new column named `day`.
				* Average Age: The column `age` is given as a range with a type of string. It is converted to an average age column considering the lower and upper age values with a type of float.
		* Dependent Variable
			* Click-Through Rate (CTR): As CTR is one of the most common metrics to measure the success of an online advertising campaign, I chose to use it as an indicator of the advertisements' success.
			* CTR is calculated for every advertisement by dividing the number of clicks by the number of impressions.
	* Predict
		* GradientBoostingRegressor is used from Python scikit-learn library as it is a powerful model by combining several weak methods.

### How to bring the model to production
* The CTR predictor is designed to give insights about the potential success of the advertisement with the related configuration in terms of target age, gender, interests and the publish date. 
* It can be used by advertisement platforms like Facebook to estimate CTR performance of the advertisements. The use cases may be
    * Given as a service to the advertisers, they become able to comprehend (i) if the ad target fits the nature of their product before publishing it and (ii) if it is the right time (in terms of week-days) to publish their ad. In this case, the advertisement platform shows the CTR prediction for related ad right at the ad configuration screen before the advertiser publish it. Taking advantage of such insight, the advertiser can find the optimal configuration to maximize CTR.
    * The ad platform can bid the related ad with respect to its CTR prediction and increase its visibility. In this case, it is an internal service of the ad platform that is deployed as one of the components for the bid assignment. There may be many other components interesting individual user's profile for bidding.

### Installing & Running
* Run `pip install -r requirements.txt`, to install dependencies.
* Run `python predictor.py`, to make predictions.
* The results are saved under `results` folder.

### Notes

* For coding, PEP8 convention is followed.