# Priority Queue experiment

## The task: 

Given a set of n mortgage candidates which includes relevant details, prioritize
them by the chances that they'll be a successful sale.

## The data

### Phase 1

1. Datasets with strict pairs, where all the information is identical except for
   the applicant name which will have a either typically Black and typically
   White name.
2. Several templates that have different presentation of the application data
3. Stick to a few locales that are more race-neutral (figure out what that
   means.)

### Phase 2

1. Draw all the dataset points through some kind of random process.
2. Stay with the race-neutral locales
3. Train a regression model to predict order and control for everything to
   identify the effect of the name.

### Phase 3

1. Use the model's understanding of locale as another thing to test using census
   data, does the model racialize the location as well.

## The measure

### Need to read up on this -- what does the statistics of ordering look like?
### How do we determine the effect of predictors on the ordering?
