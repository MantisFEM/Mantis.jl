#!/usr/bin/env bash

set -euo pipefail

############################################################################################
#                                           Info                                           #
############################################################################################

# This hook will enforce the commit message to follow the `Conventional Commits`
# specifications. You can find more information about them in the following link:
#   https://www.conventionalcommits.org/en/v1.0.0/

# Note: to run this script you first need to set is as executable. You can do that by
# running the command: "chmod +x commit-msg" or something equivalent in your OS.

############################################################################################
#                                      Configuration                                       #
############################################################################################

# Allowed commit types
types=(
	"bench"
	"chore"
	"ci"
	"doc"
	"feat"
	"fix"
	"perf"
	"refactor"
	"style"
	"test"
	"revert"
	"merge"
)
# Maximum title length (50 is the recommended value)
max_length=50
# Whether to allow `revert: ` commits.
revert=true
# Whether to enforce future tense for action
future=true

############################################################################################
#                                         Helpers                                          #
############################################################################################

# Message helpers

DEF='\033[0m' # Default colour
RED='\033[0;31m'
YELLOW='\033[1;33m'

echo_error(){
	echo -e "\n${RED}ERROR: ${DEF}$1" >&2
	echo -e "\nMake sure the message sticks to the following rules:\n"
	echo -n "1. The message has the structure: '<type>: <text>',"
	echo  " or for reverts: 'revert: <original_type>: <text>'."
	echo -e "   In both cases you can write '<type>!: <text>' for breaking changes.\n"
	echo -e "2. The message has one of the allowed types below:\n"
	echo -n >&2 "   "
	type_list=""
	for i in "${!types[@]}"; do
		type_list+="${types[$i]}"
		if [ $i -lt $((${#types[@]} - 1)) ]; then
			type_list+=", "
		fi
	done
	echo $type_list
	echo -e "\n3. The first word of "'<text>'" should contain only lower-case letters."
	exit 1
}

echo_warning(){
	echo -e "\n${YELLOW}WARNING: ${DEF}$1" >&2
}

# Validity helpers

warn_breaking_change() {
	echo_warning "You are creating a breaking change commit."
	echo "If this is not correct, you can change the message by running 'git commit --amend'."
}

check_length(){
	matched_message="$1"
	length="${#matched_message}"
	if (( length > $max_length )); then
		echo_error "Title exceeds allowed length of $max_length characters."
	fi
}

check_future() {
	past_tense="([a-z]+ed$)"
	indicative="([a-z]+s$)"
	gerund="([a-z]+ing$)"
	if [[ $1 =~ $past_tense || $1 =~ $indicative || $1 =~ $gerund ]]; then
		echo_error "Action verb should be in future tense. Got verb '$1'."
	fi
}

############################################################################################
#                                        Validation                                        #
############################################################################################

validate_message() {
    local msg="$1"
    # Check if commit message falls under one of the other types.
    for commit_type in "${types[@]}"
    do
        regex="^($commit_type)(\(.+\))?(!?)(: )([a-z]+)(.*)"
        if [[ $msg =~ $regex ]]; then
            check_length "${BASH_REMATCH[0]}"
            # Run revert commit logic.
            if [[ $commit_type == "revert" ]]; then
                # Check if revert commits are allowed.
                if ! $revert; then
                    echo_error "'revert' is set to false in $config_file_name."
                fi

                # Create the revert specific regular expression.
                regex="^(revert)(!?)(: )([a-z]+)(\(.+\))?(!?)(: )([a-z]+)(.*)"
                # Check if the old commit type is correct.
                if [[ $msg =~ $regex ]]; then
                    for old_commit_type in "${types[@]}"
                    do
                        if [[ ${BASH_REMATCH[4]} == $old_commit_type ]]; then
                            # Check for breaking change
                            if [[ -n "${BASH_REMATCH[2]}" ]]; then
                                warn_breaking_change
                            fi

                            # Check action verb tense
                            if $future; then
                                check_future "${BASH_REMATCH[8]}"
                            fi

                            exit 0
                        fi
                    done
                fi

                echo_error "Wrong type of 'revert' message."
            else
                # Check for breaking change
                if [[ -n "${BASH_REMATCH[3]}" ]]; then
                    warn_breaking_change
                fi

                # Check action verb tense
                if $future; then
                    check_future "${BASH_REMATCH[5]}"
                fi

                exit 0
            fi
        fi
    done

    echo_error "Invalid commit message: $msg"
}

if [[ $# -lt 1 ]]; then
  echo_error "Usage: $0 <commit-message-file>"
fi

# Assign commit message title to a variable
msg=$(head -1 $1)

validate_message "$msg"
