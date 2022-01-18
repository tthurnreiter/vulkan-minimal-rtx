for f in $(grep map_K[da] sponza.fixed.mtl | cut -f2 -d ' ' | sort | uniq ) ; do
	echo $f
    ( for channel in $(for expr in $(convert "$f" -resize 1x1 txt:- | tail -n1 | sed -e 's/.*srgb(//' -e 's:,:/255 :g' -e 's:):/255:') ; do calc -p "round($expr,3)"; done) ; do echo -n "$channel "; done; echo ) > $f.avg
done


