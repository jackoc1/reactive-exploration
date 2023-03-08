while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--config)
      CONFIG="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

LOG_DIR=/home/jaoc1/fyp/curiosity_outputs/logs

cd "/home/jaoc1/fyp/reactive-exploration"

conda run -n reactive_exploration python main.py --multirun env_params="$CONFIG".yaml seed=1,2,3 \
	1>$LOG_DIR/$CONFIG\_1_$(date +%s)_out.log 2>$LOG_DIR/$CONFIG\_1_$(date +%s)_error.log

# conda run -n reactive_exploration python main.py env_params="$CONFIG".yaml seed=2 \
#	1>$LOG_DIR/$CONFIG\_2_$(date +%s)_out.log 2>$LOG_DIR/$CONFIG\_2_$(date +%s)_error.log &

#conda run -n reactive_exploration python main.py env_params="$CONFIG".yaml seed=3 \
#	1>$LOG_DIR/$CONFIG\_3_$(date +%s)_out.log 2>$LOG_DIR/$CONFIG\_3_$(date +%s)_error.log &
