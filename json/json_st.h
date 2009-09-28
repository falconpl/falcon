/*
   FALCON - The Falcon Programming Language.
   FILE: json_st.h

   JSON module - string table.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin:  Sun, 27 Sep 2009 20:20:37 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef flc_json_st_H
#define flc_json_st_H

#include <falcon/message_defs.h>

FAL_MODSTR( json_msg_non_codeable, "Given object cannot be rendered as json string" );
FAL_MODSTR( json_msg_non_decodable, "Data is not in json format" );
FAL_MODSTR( json_msg_non_apply, "JSON Data not applicable to given object." );

#endif

/* end of json_st.h */
