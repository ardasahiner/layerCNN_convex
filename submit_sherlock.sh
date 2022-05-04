count=1

for ((lr0=-3; lr0<=-1; lr0+=1))
do
	for ((lr1=-3; lr1<=-1; lr1+=1))
	do
		for ((lr2=-4; lr2<=-1; lr2+=1))
		do
			for ((lr3=-4; lr3<=-2; lr3+=1))
			do
				for ((lr4=-4; lr4<=-2; lr4+=1))
				do
					for ((mse=0; mse<=1; mse+=1))
					do
						for ((hinge=0; hinge<=2; hinge+=1))
						do
							if [ $hinge -gt 0 ]
							then
								for ((hinge0=-4; hinge0<=-1; hinge0+=1))
								do
									for ((hinge1=-4; hinge1<=-1; hinge1+=1))
									do
										for ((hinge2=-4; hinge2<=-2; hinge2+=1))
										do
											for ((hinge3=-4; hinge3<=-2; hinge3+=1))
											do
												for ((hinge4=-4; hinge4<=-2; hinge4+=1))
												do

													command="sbatch --mem=64G -N 1 -n 1 -p owners --gres gpu:1 -o "layercnn_$count" --time=12:00:00 run_sherlock.sh $lr0 $lr1 $lr2 $lr3 $lr4 $mse $hinge $hinge0 $hinge1 $hinge2 $hinge3 $hinge4"
													$command
													count=`expr $count + 1`

												done
											done
										done
									done
								done
							else
								command="sbatch --mem=64G -N 1 -n 1 -p owners --gres gpu:1 -o "layercnn_$count" --time=12:00:00 run_sherlock.sh $lr0 $lr1 $lr2 $lr3 $lr4 $mse $hinge"
								$command
								count=`expr $count + 1`

							fi
						done
					done
				done
			done
		done
	done
done

