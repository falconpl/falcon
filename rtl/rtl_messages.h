/*
   FALCON - The Falcon Programming Language.
   FILE: rtl_messages.h

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab feb 24 2007

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Short description
*/

#ifndef flc_rtl_messages_H
#define flc_rtl_messages_H

namespace Falcon {
namespace msg {
   const int rtl_array_missing = 0;
   const int rtl_inv_startend = 1;
   const int rtl_arrpar1 = 2;
   const int rtl_arrpar2 = 3;
   const int rtl_arrpar3 = 4;
   const int rtl_scan_end = 5;
   const int rtl_array_first = 6;
   const int rtl_second_call = 7;
   const int rtl_need_two_arr = 8;
   const int rtl_idx_not_num = 9;
   const int rtl_start_outrange = 10;
   const int rtl_cmdp_0 = 11;
   const int rtl_emptyarr = 12;
   const int rtl_iterator_not_found = 13;
   const int rtl_invalid_iter = 14;
   const int rtl_sender_not_object = 15;
   const int rtl_marshall_not_cb = 16;
   const int rtl_invalid_path = 17;
   const int rtl_invalid_uri = 18;
}
}

#endif

/* end of rtl_messages.h */
