# Feature Identification And Address Mapping

Application that uses machine learning to find and identify residential addresses with specific feature based on satellite imagery.

### Development and Prerequisites

- docker and docker-compose


### Running the project for the first time:
- Clone the code:
```bash
> git clone git@github.com:msmsh83/featureextraction.git
```
- Run the project:
```bash
> cd featureextraction
> docker-compose up
```

- Run the django migrations to create the database and the database models.
```bash
> docker-compose run web api/manage.py migrate
```

- Create a django super user:
```bash
> docker-compose run web api/manage.py createsuperuser
```

- Import the training data sets into the project directory under /api/ml/training_images and the test data sets under /api/ml/test_images

- Retraing the machine learning models:
```bash
> docker-compose run web api/manage.py retrain
```

and then:

```bash
> docker-compose run web api/manage.py retrain --size=small
```


### Running the finding featuress script:
This script requires the access to the imagery source APIs and the Google Maps API, for that we need to set those API Keys in the .env file in the root directory of the project:

```
GOOGLE_MAPS_KEY=get_your_api_key_from_your_google_cloud_account
MAPBOX_TOKEN=get_your_token_from_your_mapbox_account
```

The script also assumes that your database is fed with the sliced polygons of the filters, for getting those filters, you can either get a backup, or get the original KML files of all the states, and run the slicing script on them on your local using:
```bash
> docker-compose run web api/manage.py processKml --input='the kml file path.'
```

And then you can run the features finding script:

```bash
> docker-compose run web api/manage.py findFeatures
```