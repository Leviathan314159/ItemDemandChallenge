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

# Read in the data -----------------------

base_folder <- "ItemDemandChallenge/"
demand_train <- vroom(paste0(base_folder, "train.csv"))
demand_test <- vroom(paste0(base_folder, "test.csv"))
filtered_demand <- demand_train |> filter(store == 2, item == 21)
filtered_demand

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