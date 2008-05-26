/*
   FALCON - The Falcon Programming Language.
   FILE: rtl_messages.h

   String table used for RTL messages.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab feb 24 2007

   -------------------------------------------------------------------
   (C) Copyright 2004-2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   String table used for RTL messages.
*/
#include <falcon/message_defs.h>

FAL_MODSTR( rtl_array_missing, L"required an array, a start and an end position" );
FAL_MODSTR( rtl_inv_startend, L"invalid start/end positions" );
FAL_MODSTR( rtl_arrpar1, L"requres an array and another parameter" );
FAL_MODSTR( rtl_arrpar2, L"optional third parameter must be a number" );
FAL_MODSTR( rtl_arrpar3, L"optional fourth parameter must be a number" );
FAL_MODSTR( rtl_scan_end, L"scan end is greater than start" );
FAL_MODSTR( rtl_array_first, L"requires an array as first parameter" );
FAL_MODSTR( rtl_second_call, L"second parameter must be callable" );
FAL_MODSTR( rtl_need_two_arr, L"needs two arrays" );
FAL_MODSTR( rtl_idx_not_num, L"indexes must be numbers" );
FAL_MODSTR( rtl_start_outrange, L"start position out of range" );
FAL_MODSTR( rtl_cmdp_0, L"parameter array contains non string elements" );
FAL_MODSTR( rtl_emptyarr, L"parameter array is empty" );
FAL_MODSTR( rtl_iterator_not_found, L"\"Iterator\" class not found in VM" );
FAL_MODSTR( rtl_invalid_iter, L"Given item is not a valid iterator for the collection" );
FAL_MODSTR( rtl_sender_not_object, L"Sender is not an object" );
FAL_MODSTR( rtl_marshall_not_cb, L"Marshalled event name must be a string as first element in the given array" );
FAL_MODSTR( rtl_invalid_path, L"Invalid path" );
FAL_MODSTR( rtl_invalid_uri, L"Invalid URI" );

/* end of rtl_messages.h */
