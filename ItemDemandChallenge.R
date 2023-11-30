# This file is for R analysis for the Item Demand Challenge.

# Libraries ---------------------
library(tidymodels)
library(vroom)
library(tidyverse)
library(embed)
library(ranger)
library(discrim)
library(themis) # for SMOTE
# library(bonsai)
# library(lightgbm)
library(modeltime)
library(timetk)
library(forecast)

# Read in the data -----------------------

base_folder <- "ItemDemandChallenge/"
demand_train <- vroom(paste0(base_folder, "train.csv"))
demand_test <- vroom(paste0(base_folder, "test.csv"))
filtered_demand <- demand_train |> filter(store == 2, item == 21)
filtered_demand
filtered_demand_2 <- demand_train |> filter(store == 6, item == 3)
filtered_demand_2
filtered_test <- demand_test |> filter(store == 2, item == 21)
filtered_test_2 <- demand_test |> filter(store == 6, item == 3)

# EDA -----------------
glimpse(demand_train)

sales_by_store <- demand_train |> group_by(store) |> summarize("total_sales" = sum(sales))
sales_by_store
sales_by_item <- demand_train |> group_by(item) |> summarize("total_sales" = sum(sales))
sales_by_item

ggplot(data = sales_by_store, aes(x = store, y = total_sales)) + geom_col()
ggplot(data = sales_by_item, aes(x = item, y = total_sales)) + geom_col()

demand_train |> filter(item == 1) |> ggplot(aes(x = date, y = sales, color = as.factor(store))) + geom_point()
demand_train |> filter(item == 3) |> ggplot(aes(x = date, y = sales, color = as.factor(store))) + geom_point()
demand_train |> filter(item == 7) |> ggplot(aes(x = date, y = sales, color = as.factor(store))) + geom_point()
demand_train |> filter(item == 15) |> ggplot(aes(x = date, y = sales, color = as.factor(store))) + geom_point()

# Recipes ----------------
# Filter to only one combination of store and item. Then eventually loop
# through each combination of store and item, applying the same process
# to each combination.
flat_demand_recipe <- recipe(sales ~ date, data = filtered_demand) |> 
  step_date(date, features = c("dow", "month", "year", "doy"))
flat_demand_recipe |> prep() |> bake(new_data = filtered_demand)

cyclic_demand_recipe <- recipe(sales ~ date, data = filtered_demand) |> 
  step_date(date, features = "doy") |> 
  step_range(date_doy, min = 0, max = pi) |> 
  step_mutate(sinDOY = sin(date_doy), cosDOY = cos(date_doy))
cyclic_demand_recipe |> prep() |> bake(new_data = filtered_demand)

combined_demand_recipe <- recipe(sales ~ date, data = filtered_demand) |> 
  step_date(date, features = c("dow", "month", "year", "doy")) |> 
  step_range(date_doy, min = 0, max = pi) |> 
  step_mutate(sinDOY = sin(date_doy), cosDOY = cos(date_doy))
combined_demand_recipe |> prep() |> bake(new_data = filtered_demand)

arima_recipe_1 <- recipe(sales ~ date, data = filtered_demand) |> 
  step_date(date, features = c("dow", "month", "year", "doy"))
arima_recipe_2 <- recipe(sales ~ date, data = filtered_demand_2) |> 
  step_date(date, features = c("dow", "month", "year", "doy"))

# Flat Random Forest -----------------
flat_forest_model <- rand_forest(mtry = tune(),
                             min_n = tune(),
                             trees = 1000) |>
  set_engine("ranger") |>
  set_mode("regression")

# Create a workflow using the model and recipe
flat_forest_wf <- workflow() |>
  add_model(flat_forest_model) |>
  add_recipe(flat_demand_recipe)

# Set up the grid with the tuning values
flat_forest_grid <- grid_regular(mtry(range = c(1, (length(filtered_demand)-1))), min_n(), levels = 5)

# Set up the K-fold CV
flat_forest_folds <- vfold_cv(data = filtered_demand, v = 10, repeats = 1)

# Find best tuning parameters
flat_forest_cv_results <- flat_forest_wf |>
  tune_grid(resamples = flat_forest_folds,
            grid = flat_forest_grid,
            metrics = metric_set(smape))

# Find out the best tuning parameters
flat_forest_tuning <- collect_metrics(flat_forest_cv_results)
flat_forest_tuning

flat_forest_best_tuning <- flat_forest_tuning |> slice_min(order_by = mean)
flat_forest_best_tuning

# Cyclic Random Forest ---------------
cyclic_forest_model <- rand_forest(mtry = tune(),
                                 min_n = tune(),
                                 trees = 1000) |>
  set_engine("ranger") |>
  set_mode("regression")

# Create a workflow using the model and recipe
cyclic_forest_wf <- workflow() |>
  add_model(cyclic_forest_model) |>
  add_recipe(cyclic_demand_recipe)

# Set up the grid with the tuning values
cyclic_forest_grid <- grid_regular(mtry(range = c(1, (length(filtered_demand)-1))), min_n(), levels = 5)

# Set up the K-fold CV
cyclic_forest_folds <- vfold_cv(data = filtered_demand, v = 10, repeats = 1)

# Find best tuning parameters
cyclic_forest_cv_results <- cyclic_forest_wf |>
  tune_grid(resamples = cyclic_forest_folds,
            grid = cyclic_forest_grid,
            metrics = metric_set(smape))

# Find out the best tuning parameters
cyclic_forest_tuning <- collect_metrics(cyclic_forest_cv_results)
cyclic_forest_tuning

cyclic_forest_best_tuning <- cyclic_forest_tuning |> slice_min(order_by = mean)
cyclic_forest_best_tuning

# Combined Random Forest -------------
# Set the model
combined_forest_model <- rand_forest(mtry = tune(),
                                 min_n = tune(),
                                 trees = 1000) |>
  set_engine("ranger") |>
  set_mode("regression")

# Create a workflow using the model and recipe
combined_forest_wf <- workflow() |>
  add_model(combined_forest_model) |>
  add_recipe(combined_demand_recipe)

# Set up the grid with the tuning values
combined_forest_grid <- grid_regular(mtry(range = c(1, (length(filtered_demand)-1))), min_n(), levels = 5)

# Set up the K-fold CV
combined_forest_folds <- vfold_cv(data = filtered_demand, v = 10, repeats = 1)

# Find best tuning parameters
combined_forest_cv_results <- combined_forest_wf |>
  tune_grid(resamples = combined_forest_folds,
            grid = combined_forest_grid,
            metrics = metric_set(smape))

# Find out the best tuning parameters
combined_forest_tuning <- collect_metrics(combined_forest_cv_results)
combined_forest_tuning

combined_forest_best_tuning <- combined_forest_tuning |> slice_min(order_by = mean)
combined_forest_best_tuning

# Exponential Smoothing Panel Plots --------------------------

# Store_Item 1
# Define CV split
exp_cv_split_1 <- time_series_split(filtered_demand, assess="3 months", cumulative = TRUE)
exp_plan_plot_1 <- exp_cv_split_1 |> 
  tk_time_series_cv_plan() |> data.frame() |> 
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)
exp_plan_plot_1

# Set model
exp_sm_model_1 <- exp_smoothing() |> 
  set_engine("ets") |> 
  fit(sales ~ date, data = training(exp_cv_split_1))

# CV for model tuning
exp_cv_results_1 <- modeltime_calibrate(exp_sm_model_1, 
                                        new_data = testing(exp_cv_split_1))
# Visualize CV results
exp_forecast_plot_1 <- exp_cv_results_1 |> modeltime_forecast(new_data = testing(exp_cv_split_1),
                                       actual_data = filtered_demand) |> 
  plot_modeltime_forecast(.interactive = TRUE)
exp_forecast_plot_1

# Evaluate accuracy
exp_cv_results_1 |> modeltime_accuracy() |> 
  table_modeltime_accuracy(.interactive = FALSE)

# # Refit to all data then forecast
# exp_fullfit_1 <- exp_cv_results_1 |> 
#   modeltime_refit(data = filtered_demand)
# exp_preds_1 <- exp_fullfit_1 |> 
#   modeltime_forecast(h = "3 months") |> 
#   rename(date = .index, sales = .value) |> 
#   select(date, sales) |> 
#   full_join(., y = test, by = "date") |> 
#   select(id, sales)

# Store_Item 2
# Define CV split
exp_cv_split_2 <- time_series_split(filtered_demand, assess="3 months", cumulative = TRUE)
exp_plan_plot_2 <- exp_cv_split_2 |> 
  tk_time_series_cv_plan() |> data.frame() |> 
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)
exp_plan_plot_2

# Set model
exp_sm_model_2 <- exp_smoothing() |> 
  set_engine("ets") |> 
  fit(sales ~ date, data = training(exp_cv_split_2))

# CV for model tuning
exp_cv_results_2 <- modeltime_calibrate(exp_sm_model_2, 
                                        new_data = testing(exp_cv_split_2))
# Visualize CV results
exp_forecast_plot_2 <- exp_cv_results_2 |> modeltime_forecast(new_data = testing(exp_cv_split_2),
                                                              actual_data = filtered_demand_2) |> 
  plot_modeltime_forecast(.interactive = TRUE)
exp_forecast_plot_2

# Evaluate accuracy
exp_cv_results_2 |> modeltime_accuracy() |> 
  table_modeltime_accuracy(.interactive = FALSE)

# 4 Panel plot using plots from Store_Item 1 and Store_Item 2
panel_plot <- plotly::subplot(exp_plan_plot_1, exp_plan_plot_2, exp_forecast_plot_1, exp_forecast_plot_2, nrows = 2)
panel_plot

# ARIMA Models ----------------------------------

# Store_Item 1
# Define CV split
arima_split_1 <- time_series_split(filtered_demand, assess="3 months", cumulative = TRUE)
arima_plan_plot_1 <- arima_split_1 |> 
  tk_time_series_cv_plan() |> data.frame() |> 
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)
arima_plan_plot_1

# Set model
arima_model_1 <- arima_reg(#seasonal_period = 365,
    non_seasonal_ar=5, # default max p to tune3
    non_seasonal_ma=5, # default max q to tune4
    seasonal_ar=2, # default max P to tune5
    seasonal_ma=2, #default max Q to tune6
    non_seasonal_differences=2, # default max d to tune
    seasonal_differences=2 #default max D to tune
  ) |> 
  set_engine("auto_arima")

# Set the workflow
arima_wf <- workflow() |> 
  add_recipe(arima_recipe_1) |> 
  add_model(arima_model_1) |> 
  fit(data = training(arima_split_1))

# CV for model tuning
arima_results_1 <- modeltime_calibrate(arima_wf, 
                                        new_data = testing(arima_split_1))
# Visualize CV results
arima_cv_predict_plot_1 <- arima_results_1 |> modeltime_forecast(new_data = testing(arima_split_1),
                                                              actual_data = filtered_demand) |> 
  plot_modeltime_forecast(.interactive = TRUE)
arima_cv_predict_plot_1

# Evaluate accuracy
arima_results_1 |> modeltime_accuracy() |> 
  table_modeltime_accuracy(.interactive = FALSE)

# # Refit to all data then forecast
arima_fullfit_1 <- arima_results_1 |> 
  modeltime_refit(data = filtered_demand)
arima_preds_1 <- arima_fullfit_1 |> 
  modeltime_forecast(h = "3 months", 
                     new_data = filtered_test, 
                     actual_data = filtered_demand) |> 
  rename(date = .index, sales = .value) |> 
  select(date, sales) |> 
  full_join(., y = filtered_test, by = "date") |> 
  select(id, sales)
first_index <- length(filtered_demand$date) + 1
true_id_1 <- arima_preds_1$id[first_index:length(arima_preds_1$id)]
true_sales_1 <- arima_preds_1$sales[first_index:length(arima_preds_1$sales)]
true_arima_preds_1 <- data.frame("id" = true_id_1, "sales" = true_sales_1)
true_arima_preds_1

# Create 3 month forecast plot
arima_forecast_plot_1 <- arima_results_1 |> 
  modeltime_forecast(h = "3 months", 
                     new_data = filtered_test, 
                     actual_data = filtered_demand) |> 
  slice_tail(n = length(true_arima_preds_1$id)) |> 
  plot_modeltime_forecast(.interactive = TRUE)
arima_forecast_plot_1

# Store_Item 2
# Define CV split
arima_split_2 <- time_series_split(filtered_demand_2, assess="3 months", cumulative = TRUE)
arima_plan_plot_2 <- arima_split_2 |> 
  tk_time_series_cv_plan() |> data.frame() |> 
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)
arima_plan_plot_2

# Set model
arima_model_2 <- arima_reg(#seasonal_period = 365,
  non_seasonal_ar=5, # default max p to tune3
  non_seasonal_ma=5, # default max q to tune4
  seasonal_ar=2, # default max P to tune5
  seasonal_ma=2, #default max Q to tune6
  non_seasonal_differences=2, # default max d to tune
  seasonal_differences=2 #default max D to tune
) |> 
  set_engine("auto_arima")

# Set the workflow
arima_wf_2 <- workflow() |> 
  add_recipe(arima_recipe_2) |> 
  add_model(arima_model_2) |> 
  fit(data = training(arima_split_2))

# CV for model tuning
arima_results_2 <- modeltime_calibrate(arima_wf_2, 
                                       new_data = testing(arima_split_2))
# Visualize CV results
arima_cv_predict_plot_2 <- arima_results_2 |> modeltime_forecast(new_data = testing(arima_split_2),
                                                                 actual_data = filtered_demand_2) |> 
  plot_modeltime_forecast(.interactive = TRUE)
arima_cv_predict_plot_2

# Evaluate accuracy
arima_results_2 |> modeltime_accuracy() |> 
  table_modeltime_accuracy(.interactive = FALSE)

# # Refit to all data then forecast
arima_fullfit_2 <- arima_results_2 |> 
  modeltime_refit(data = filtered_demand_2)
arima_preds_2 <- arima_fullfit_2 |> 
  modeltime_forecast(h = "3 months", 
                     new_data = filtered_test_2, 
                     actual_data = filtered_demand_2) |> 
  rename(date = .index, sales = .value) |> 
  select(date, sales) |> 
  full_join(., y = filtered_test, by = "date") |> 
  select(id, sales)
first_index <- length(filtered_demand_2$date) + 1
true_id_2 <- arima_preds_2$id[first_index:length(arima_preds_2$id)]
true_sales_2 <- arima_preds_2$sales[first_index:length(arima_preds_2$sales)]
true_arima_preds_2 <- data.frame("id" = true_id_2, "sales" = true_sales_2)
true_arima_preds_2

# Create 3 month forecast plot
arima_forecast_plot_2 <- arima_results_2 |> 
  modeltime_forecast(h = "3 months", 
                     new_data = filtered_test_2, 
                     actual_data = filtered_demand_2) |> 
  slice_tail(n = length(true_arima_preds_2$id)) |> 
  plot_modeltime_forecast(.interactive = TRUE)
arima_forecast_plot_2

# Create the 4-panel plot
arima_panel_plot <- plotly::subplot(arima_cv_predict_plot_1, 
                                      arima_cv_predict_plot_2,
                                      arima_forecast_plot_1,
                                      arima_forecast_plot_2,
                                      nrows = 2)
arima_panel_plot


# Facebook's Prophet Model ---------------------------

# Store_Item 1
# Define CV split
prophet_split_1 <- time_series_split(filtered_demand, assess="3 months", cumulative = TRUE)
prophet_plan_plot_1 <- prophet_split_1 |> 
  tk_time_series_cv_plan() |> data.frame() |> 
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)
prophet_plan_plot_1

# Set model
prophet_model_1 <- prophet_reg() |> 
  set_engine(engine = "prophet") |> 
  fit(sales ~ date, data = training(prophet_split_1))

# CV for model tuning
prophet_results_1 <- modeltime_calibrate(prophet_model_1, 
                                        new_data = testing(prophet_split_1))
# Visualize CV results
prophet_cv_predict_plot_1 <- prophet_results_1 |> modeltime_forecast(new_data = testing(prophet_split_1),
                                                              actual_data = filtered_demand) |> 
  plot_modeltime_forecast(.interactive = TRUE)
prophet_cv_predict_plot_1

# Evaluate accuracy
prophet_results_1 |> modeltime_accuracy() |> 
  table_modeltime_accuracy(.interactive = FALSE)

# # Refit to all data then forecast
prophet_fullfit_1 <- prophet_results_1 |> 
  modeltime_refit(data = filtered_demand)
prophet_preds_1 <- prophet_fullfit_1 |> 
  modeltime_forecast(h = "3 months", 
                     new_data = filtered_test, 
                     actual_data = filtered_demand) |> 
  rename(date = .index, sales = .value) |> 
  select(date, sales) |> 
  full_join(., y = filtered_test, by = "date") |> 
  select(id, sales)
first_index <- length(filtered_demand$date) + 1
true_id_1 <- prophet_preds_1$id[first_index:length(prophet_preds_1$id)]
true_sales_1 <- prophet_preds_1$sales[first_index:length(prophet_preds_1$sales)]
true_prophet_preds_1 <- data.frame("id" = true_id_1, "sales" = true_sales_1)
true_prophet_preds_1

# Create 3 month forecast plot
prophet_forecast_plot_1 <- prophet_results_1 |> 
  modeltime_forecast(h = "3 months", 
                    new_data = filtered_test, 
                    actual_data = filtered_demand) |> 
  slice_tail(n = length(true_prophet_preds_1$id)) |> 
  plot_modeltime_forecast(.interactive = TRUE)
prophet_forecast_plot_1

# Store_Item 2
# Define CV split
prophet_split_2 <- time_series_split(filtered_demand_2, assess="3 months", cumulative = TRUE)
prophet_plan_plot_2 <- prophet_split_2 |> 
  tk_time_series_cv_plan() |> data.frame() |> 
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)
prophet_plan_plot_2

# Set model
prophet_model_2 <- prophet_reg() |> 
  set_engine(engine = "prophet") |> 
  fit(sales ~ date, data = training(prophet_split_2))

# CV for model tuning
prophet_results_2 <- modeltime_calibrate(prophet_model_2, 
                                         new_data = testing(prophet_split_2))
# Visualize CV results
prophet_cv_predict_plot_2 <- prophet_results_2 |> modeltime_forecast(new_data = testing(prophet_split_2),
                                                                   actual_data = filtered_demand_2) |> 
  plot_modeltime_forecast(.interactive = TRUE)
prophet_cv_predict_plot_2

# Evaluate accuracy
prophet_results_2 |> modeltime_accuracy() |> 
  table_modeltime_accuracy(.interactive = FALSE)

# # Refit to all data then forecast
prophet_fullfit_2 <- prophet_results_2 |> 
  modeltime_refit(data = filtered_demand_2)
prophet_preds_2 <- prophet_fullfit_2 |> 
  modeltime_forecast(h = "3 months", 
                     new_data = filtered_test_2, 
                     actual_data = filtered_demand_2) |> 
  rename(date = .index, sales = .value) |> 
  select(date, sales) |> 
  full_join(., y = filtered_test_2, by = "date") |> 
  select(id, sales)
first_index <- length(filtered_demand_2$date) + 1
true_id_2 <- prophet_preds_2$id[first_index:length(prophet_preds_2$id)]
true_sales_2 <- prophet_preds_2$sales[first_index:length(prophet_preds_2$sales)]
true_prophet_preds_2 <- data.frame("id" = true_id_2, "sales" = true_sales_2)
true_prophet_preds_2

# Create 3 month forecast plot
prophet_forecast_plot_2 <- prophet_results_2 |> 
  modeltime_forecast(h = "3 months", 
                     new_data = filtered_test_2, 
                     actual_data = filtered_demand_2) |> 
  slice_tail(n = length(true_prophet_preds_2$id)) |> 
  plot_modeltime_forecast(.interactive = TRUE)
prophet_forecast_plot_2

# Create the 4-panel plot
prophet_panel_plot <- plotly::subplot(prophet_cv_predict_plot_1, 
                              prophet_cv_predict_plot_2,
                              prophet_forecast_plot_1,
                              prophet_forecast_plot_2,
                              nrows = 2)
prophet_panel_plot


# Implement Later -------------------------
# Wrap the prophet model (and any other model for that matter)
#   in a function that will do the modeling. Use a filtered training and
#   a filtered test datasets as the two inputs for the function. Then
#   output a dataframe containing the prediction dataframes as desired.
# After doing this, write a loop that will filter each combination of 
#   store_item and run the modeling for each combination.