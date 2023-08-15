task_name = 'Test'
city = 'PHO'  # PHO, NYC, SIN
gpuId = "cuda:0"

enable_random_mask = True
mask_prop = 0.1
enable_enhance_user = True
enable_ssl = True  # whether enable contrastive learning
enable_distance_sample = False  # whether sample negative samples by distance
neg_sample_count = 5
neg_weight = 1

enable_dynamic_day_length = False
sample_day_length = 14  # range [3,14]

lr = 1e-4
epoch = 25
if city == 'SIN':
    embed_size = 60
    run_times = 3
elif city == 'NYC':
    embed_size = 40
    run_times = 3
elif city == 'PHO':
    embed_size = 60
    run_times = 5

output_file_name = f'{task_name} {city}' + "_epoch" + str(epoch)

if enable_dynamic_day_length:
    output_file_name = output_file_name + f"_DynamicDay{sample_day_length}"
else:
    output_file_name = output_file_name + "_StaticDay7"

if enable_random_mask:
    output_file_name = output_file_name + "_" + "Mask"
else:
    output_file_name = output_file_name + "_" + "NoMask"

if enable_enhance_user:
    output_file_name = output_file_name + "_" + "Enhance"
else:
    output_file_name = output_file_name + "_" + "NoEnhance"

if enable_ssl:
    if enable_distance_sample:
        output_file_name = output_file_name + "_" + "SSL" + "_" + "DistanceNegCount" + str(neg_sample_count)
    else:
        output_file_name = output_file_name + "_" + "SSL" + "_" + "NegCount" + str(neg_sample_count)
else:
    output_file_name = output_file_name + "_" + "NoSSL"

output_file_name = output_file_name + '_embeddingSize' + str(embed_size)
