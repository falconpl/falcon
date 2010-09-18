/*
 * encoder_unicode.cpp
 *
 *  Created on: 12/set/2010
 *      Author: gian
 */

#include "hpdf_conf.h"
#include "hpdf_utils.h"
#include "hpdf_encoder.h"
#include "hpdf.h"

#include <string.h>

namespace Falcon{ namespace Mod { namespace hpdf {

HPDF_ByteType
HPDF_UTF8Encoder_ByteType  (HPDF_Encoder        encoder,
                            HPDF_ParseText_Rec  *state)
{
    HPDF_CMapEncoderAttr attr = (HPDF_CMapEncoderAttr)encoder->attr;

    HPDF_PTRACE ((" HPDF_UTF8Encoder_ByteType\n"));
    state->byte_type = HPDF_BYTE_TYPE_SINGLE;
    state->index++;
    return state->byte_type;
}

HPDF_UNICODE
HPDF_UTF8Encoder_ToUnicode  (HPDF_Encoder  encoder,
                             HPDF_UINT16   code)
{
    return 'a';
}


static HPDF_STATUS
UTF8_CommonInit  (HPDF_Encoder  encoder)
{
   encoder->byte_type_fn = HPDF_UTF8Encoder_ByteType;
   encoder->to_unicode_fn = HPDF_UTF8Encoder_ToUnicode;
   strcpy( encoder->name, "UTF-8" );

   return HPDF_OK;
}

/*--------------------------------------------------------------------------*/

HPDF_STATUS
HPDF_UseUnicodeEncodings   (HPDF_Doc   pdf)
{
    HPDF_Encoder encoder;
    HPDF_STATUS ret;

    if (!HPDF_HasDoc (pdf))
        return HPDF_INVALID_DOCUMENT;

    /* Horizontal unicode. */
    encoder = HPDF_BasicEncoder_New( pdf->mmgr,  HPDF_ENCODING_WIN_ANSI );
    UTF8_CommonInit( encoder );

    if ((ret = HPDF_Doc_RegisterEncoder (pdf, encoder)) != HPDF_OK)
        return ret;


    return HPDF_OK;
}

}}} // Namespace
