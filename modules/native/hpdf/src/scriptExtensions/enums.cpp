/*
 * pdf.cpp
 *
 *  Created on: 04.04.2010
 *      Author: maik
 */

/*#
   @beginmodule hpdf
*/
#include <falcon/engine.h>
#include <hpdf.h>
#include "enums.h"

namespace Falcon { namespace Ext { namespace hpdf {

 void registerEnums(Falcon::Module* self)
{
  {
    Falcon::Symbol* fclass = self->addClass( "InfoType" );
    fclass->setEnum(true);
    self->addClassProperty( fclass, "CREATION_DATE" )
      .setInteger(HPDF_INFO_CREATION_DATE).setReadOnly(true);
    self->addClassProperty( fclass, "MOD_DATE" )
      .setInteger(HPDF_INFO_MOD_DATE).setReadOnly(true);
    self->addClassProperty( fclass, "AUTHOR" )
      .setInteger(HPDF_INFO_AUTHOR).setReadOnly(true);
     self->addClassProperty( fclass, "CREATOR" )
      .setInteger(HPDF_INFO_CREATOR).setReadOnly(true);
     self->addClassProperty( fclass, "PRODUCER" )
      .setInteger(HPDF_INFO_PRODUCER).setReadOnly(true);
     self->addClassProperty( fclass, "TITLE" )
      .setInteger(HPDF_INFO_TITLE).setReadOnly(true);
     self->addClassProperty( fclass, "SUBJECT" )
      .setInteger(HPDF_INFO_SUBJECT).setReadOnly(true);
     self->addClassProperty( fclass, "KEYWORDS" )
      .setInteger(HPDF_INFO_KEYWORDS).setReadOnly(true);
  }

  {
    Falcon::Symbol* fclass = self->addClass( "PdfVer" );
    fclass->setEnum(true);
    self->addClassProperty( fclass, "VER_12" )
      .setInteger(HPDF_VER_12).setReadOnly(true);
    self->addClassProperty( fclass, "VER_13" )
      .setInteger(HPDF_VER_13).setReadOnly(true);
    self->addClassProperty( fclass, "VER_14" )
      .setInteger(HPDF_VER_14).setReadOnly(true);
    self->addClassProperty( fclass, "VER_15" )
      .setInteger(HPDF_VER_15).setReadOnly(true);
    self->addClassProperty( fclass, "VER_16" )
      .setInteger(HPDF_VER_16).setReadOnly(true);
    self->addClassProperty( fclass, "VER_17" )
      .setInteger(HPDF_VER_17).setReadOnly(true);
  }

  {
    Falcon::Symbol* fclass = self->addClass( "EncryptMode" );
    fclass->setEnum(true);
    self->addClassProperty( fclass, "R2" )
      .setInteger(HPDF_ENCRYPT_R2).setReadOnly(true);
    self->addClassProperty( fclass, "R3" )
      .setInteger(HPDF_ENCRYPT_R3).setReadOnly(true);
  }

  {
    Falcon::Symbol* fclass = self->addClass( "ColorSpace" );
    fclass->setEnum(true);
    self->addClassProperty( fclass, "DEVICE_GRAY" )
      .setInteger(HPDF_CS_DEVICE_GRAY).setReadOnly(true);
    self->addClassProperty( fclass, "DEVICE_RGB" )
      .setInteger(HPDF_CS_DEVICE_RGB).setReadOnly(true);
    self->addClassProperty( fclass, "DEVICE_CMYK" )
      .setInteger(HPDF_CS_DEVICE_CMYK).setReadOnly(true);
    self->addClassProperty( fclass, "CAL_GRAY" )
      .setInteger(HPDF_CS_CAL_GRAY).setReadOnly(true);
    self->addClassProperty( fclass, "CAL_RGB" )
      .setInteger(HPDF_CS_CAL_RGB).setReadOnly(true);
    self->addClassProperty( fclass, "LAB" )
      .setInteger(HPDF_CS_LAB).setReadOnly(true);
    self->addClassProperty( fclass, "ICC_BASED" )
      .setInteger(HPDF_CS_ICC_BASED).setReadOnly(true);
    self->addClassProperty( fclass, "SEPARATION" )
      .setInteger(HPDF_CS_SEPARATION).setReadOnly(true);
    self->addClassProperty( fclass, "DEVICE_N" )
      .setInteger(HPDF_CS_DEVICE_N).setReadOnly(true);
    self->addClassProperty( fclass, "INDEXED" )
      .setInteger(HPDF_CS_INDEXED).setReadOnly(true);
    self->addClassProperty( fclass, "PATTERN" )
      .setInteger(HPDF_CS_PATTERN).setReadOnly(true);
  }

  {
    Falcon::Symbol* fclass = self->addClass( "LineCap" );
    fclass->setEnum(true);
    self->addClassProperty( fclass, "BUTT_END" )
      .setInteger(HPDF_BUTT_END).setReadOnly(true);
    self->addClassProperty( fclass, "ROUND_END" )
      .setInteger(HPDF_ROUND_END).setReadOnly(true);
    self->addClassProperty( fclass, "PROJECTING_SCUARE_END" )
      .setInteger(HPDF_PROJECTING_SCUARE_END).setReadOnly(true);
  }

  {
    Falcon::Symbol* fclass = self->addClass( "LineJoin" );
    fclass->setEnum(true);
    self->addClassProperty( fclass, "MITER_JOIN" )
      .setInteger(HPDF_MITER_JOIN).setReadOnly(true);
    self->addClassProperty( fclass, "ROUND_JOIN" )
      .setInteger(HPDF_ROUND_JOIN).setReadOnly(true);
    self->addClassProperty( fclass, "BEVEL_JOIN" )
      .setInteger(HPDF_BEVEL_JOIN).setReadOnly(true);
  }

  {
    Falcon::Symbol* fclass = self->addClass( "TextRenderingMode" );
    fclass->setEnum(true);
    self->addClassProperty( fclass, "FILL" )
      .setInteger(HPDF_FILL).setReadOnly(true);
    self->addClassProperty( fclass, "STROKE" )
      .setInteger(HPDF_STROKE).setReadOnly(true);
    self->addClassProperty( fclass, "FILL_THEN_STROKE" )
      .setInteger(HPDF_FILL_THEN_STROKE).setReadOnly(true);
    self->addClassProperty( fclass, "INVISIBLE" )
      .setInteger(HPDF_INVISIBLE).setReadOnly(true);
    self->addClassProperty( fclass, "FILL_CLIPPING" )
      .setInteger(HPDF_FILL_CLIPPING).setReadOnly(true);
    self->addClassProperty( fclass, "STROKE_CLIPPING" )
      .setInteger(HPDF_STROKE_CLIPPING).setReadOnly(true);
    self->addClassProperty( fclass, "FILL_STROKE_CLIPPING" )
      .setInteger(HPDF_FILL_STROKE_CLIPPING).setReadOnly(true);
    self->addClassProperty( fclass, "CLIPPING" )
      .setInteger(HPDF_CLIPPING).setReadOnly(true);
  }

  {
    Falcon::Symbol* fclass = self->addClass( "WritingMode" );
    fclass->setEnum(true);
    self->addClassProperty( fclass, "HORIZONTAL" )
      .setInteger(HPDF_WMODE_HORIZONTAL).setReadOnly(true);
    self->addClassProperty( fclass, "VERTICAL" )
      .setInteger(HPDF_WMODE_VERTICAL).setReadOnly(true);
  }

  {
    Falcon::Symbol* fclass = self->addClass( "PageLayout" );
    fclass->setEnum(true);
    self->addClassProperty( fclass, "SINGLE" )
      .setInteger(HPDF_PAGE_LAYOUT_SINGLE).setReadOnly(true);
    self->addClassProperty( fclass, "ONE_COLUMN" )
      .setInteger(HPDF_PAGE_LAYOUT_ONE_COLUMN).setReadOnly(true);
    self->addClassProperty( fclass, "TWO_COLUMN_LEFT" )
      .setInteger(HPDF_PAGE_LAYOUT_TWO_COLUMN_LEFT).setReadOnly(true);
    self->addClassProperty( fclass, "TWO_COLUMN_RIGHT" )
      .setInteger(HPDF_PAGE_LAYOUT_TWO_COLUMN_RIGHT).setReadOnly(true);
  }

  {
    Falcon::Symbol* fclass = self->addClass( "PageMode" );
    fclass->setEnum(true);
    self->addClassProperty( fclass, "USE_NONE" )
      .setInteger(HPDF_PAGE_MODE_USE_NONE).setReadOnly(true);
    self->addClassProperty( fclass, "USE_OUTLINE" )
      .setInteger(HPDF_PAGE_MODE_USE_OUTLINE).setReadOnly(true);
    self->addClassProperty( fclass, "USE_THUMBS" )
      .setInteger(HPDF_PAGE_MODE_USE_THUMBS).setReadOnly(true);
    self->addClassProperty( fclass, "FULL_SCREEN" )
      .setInteger(HPDF_PAGE_MODE_FULL_SCREEN).setReadOnly(true);
  }

  {
    Falcon::Symbol* fclass = self->addClass( "PageNumStyle" );
    fclass->setEnum(true);
    self->addClassProperty( fclass, "DECIMAL" )
      .setInteger(HPDF_PAGE_NUM_STYLE_DECIMAL).setReadOnly(true);
    self->addClassProperty( fclass, "UPPER_ROMAN" )
      .setInteger(HPDF_PAGE_NUM_STYLE_UPPER_ROMAN).setReadOnly(true);
    self->addClassProperty( fclass, "LOWER_ROMAN" )
      .setInteger(HPDF_PAGE_NUM_STYLE_LOWER_ROMAN).setReadOnly(true);
    self->addClassProperty( fclass, "UPPER_LETTERS" )
      .setInteger(HPDF_PAGE_NUM_STYLE_UPPER_LETTERS).setReadOnly(true);
    self->addClassProperty( fclass, "LOWER_LETTERS" )
      .setInteger(HPDF_PAGE_NUM_STYLE_LOWER_LETTERS).setReadOnly(true);
  }

  {
    Falcon::Symbol* fclass = self->addClass( "DestinationType" );
    fclass->setEnum(true);
    self->addClassProperty( fclass, "XYZ" )
      .setInteger(HPDF_XYZ).setReadOnly(true);
    self->addClassProperty( fclass, "FIT" )
      .setInteger(HPDF_FIT).setReadOnly(true);
    self->addClassProperty( fclass, "FIT_H" )
      .setInteger(HPDF_FIT_H).setReadOnly(true);
    self->addClassProperty( fclass, "FIT_V" )
      .setInteger(HPDF_FIT_V).setReadOnly(true);
    self->addClassProperty( fclass, "FIT_R" )
      .setInteger(HPDF_FIT_R).setReadOnly(true);
    self->addClassProperty( fclass, "FIT_B" )
      .setInteger(HPDF_FIT_B).setReadOnly(true);
    self->addClassProperty( fclass, "FIT_BH" )
      .setInteger(HPDF_FIT_BH).setReadOnly(true);
    self->addClassProperty( fclass, "FIT_BV" )
      .setInteger(HPDF_FIT_BV).setReadOnly(true);
  }

  {
    Falcon::Symbol* fclass = self->addClass( "AnnotType" );
    fclass->setEnum(true);
    self->addClassProperty( fclass, "TEXT_NOTES" )
      .setInteger(HPDF_ANNOT_TEXT_NOTES).setReadOnly(true);
    self->addClassProperty( fclass, "LINK" )
      .setInteger(HPDF_ANNOT_LINK).setReadOnly(true);
    self->addClassProperty( fclass, "SOUND" )
      .setInteger(HPDF_ANNOT_SOUND).setReadOnly(true);
    self->addClassProperty( fclass, "FREE_TEXT" )
      .setInteger(HPDF_ANNOT_FREE_TEXT).setReadOnly(true);
    self->addClassProperty( fclass, "STAMP" )
      .setInteger(HPDF_ANNOT_STAMP).setReadOnly(true);
    self->addClassProperty( fclass, "SQUARE" )
      .setInteger(HPDF_ANNOT_SQUARE).setReadOnly(true);
    self->addClassProperty( fclass, "CIRCLE" )
      .setInteger(HPDF_ANNOT_CIRCLE).setReadOnly(true);
    self->addClassProperty( fclass, "STRIKE_OUT" )
      .setInteger(HPDF_ANNOT_STRIKE_OUT).setReadOnly(true);
    self->addClassProperty( fclass, "HIGHTLIGHT" )
      .setInteger(HPDF_ANNOT_HIGHTLIGHT).setReadOnly(true);
    self->addClassProperty( fclass, "UNDERLINE" )
      .setInteger(HPDF_ANNOT_UNDERLINE).setReadOnly(true);
    self->addClassProperty( fclass, "INK" )
      .setInteger(HPDF_ANNOT_INK).setReadOnly(true);
    self->addClassProperty( fclass, "FILE_ATTACHMENT" )
      .setInteger(HPDF_ANNOT_FILE_ATTACHMENT).setReadOnly(true);
    self->addClassProperty( fclass, "POPUP" )
      .setInteger(HPDF_ANNOT_POPUP).setReadOnly(true);
    self->addClassProperty( fclass, "3D" )
      .setInteger(HPDF_ANNOT_3D).setReadOnly(true);
  }

  {
    Falcon::Symbol* fclass = self->addClass( "AnnotFlgs" );
    fclass->setEnum(true);
    self->addClassProperty( fclass, "INVISIBLE" )
      .setInteger(HPDF_ANNOT_INVISIBLE).setReadOnly(true);
    self->addClassProperty( fclass, "HIDDEN" )
      .setInteger(HPDF_ANNOT_HIDDEN).setReadOnly(true);
    self->addClassProperty( fclass, "PRINT" )
      .setInteger(HPDF_ANNOT_PRINT).setReadOnly(true);
    self->addClassProperty( fclass, "NOZOOM" )
      .setInteger(HPDF_ANNOT_NOZOOM).setReadOnly(true);
    self->addClassProperty( fclass, "NOROTATE" )
      .setInteger(HPDF_ANNOT_NOROTATE).setReadOnly(true);
    self->addClassProperty( fclass, "NOVIEW" )
      .setInteger(HPDF_ANNOT_NOVIEW).setReadOnly(true);
    self->addClassProperty( fclass, "READONLY" )
      .setInteger(HPDF_ANNOT_READONLY).setReadOnly(true);
  }
  {
    Falcon::Symbol* fclass = self->addClass( "AnnotHighlightMode" );
    fclass->setEnum(true);
    self->addClassProperty( fclass, "NO_HIGHTLIGHT" )
      .setInteger(HPDF_ANNOT_NO_HIGHTLIGHT).setReadOnly(true);
    self->addClassProperty( fclass, "INVERT_BOX" )
      .setInteger(HPDF_ANNOT_INVERT_BOX).setReadOnly(true);
    self->addClassProperty( fclass, "INVERT_BORDER" )
      .setInteger(HPDF_ANNOT_INVERT_BORDER).setReadOnly(true);
    self->addClassProperty( fclass, "DOWN_APPEARANCE" )
      .setInteger(HPDF_ANNOT_DOWN_APPEARANCE).setReadOnly(true);
  }

  {
    Falcon::Symbol* fclass = self->addClass( "AnnotIcon" );
    fclass->setEnum(true);
    self->addClassProperty( fclass, "COMMENT" )
      .setInteger(HPDF_ANNOT_ICON_COMMENT).setReadOnly(true);
    self->addClassProperty( fclass, "KEY" )
      .setInteger(HPDF_ANNOT_ICON_KEY).setReadOnly(true);
    self->addClassProperty( fclass, "HELP" )
      .setInteger(HPDF_ANNOT_ICON_NOTE).setReadOnly(true);
    self->addClassProperty( fclass, "NOTE" )
      .setInteger(HPDF_ANNOT_ICON_HELP).setReadOnly(true);
    self->addClassProperty( fclass, "NEW_PARAGRAPH" )
      .setInteger(HPDF_ANNOT_ICON_NEW_PARAGRAPH).setReadOnly(true);
    self->addClassProperty( fclass, "PARAGRAPH" )
      .setInteger(HPDF_ANNOT_ICON_PARAGRAPH).setReadOnly(true);
    self->addClassProperty( fclass, "INSERT" )
      .setInteger(HPDF_ANNOT_ICON_INSERT).setReadOnly(true);
  }

  {
    Falcon::Symbol* fclass = self->addClass( "BlendMode" );
    fclass->setEnum(true);
    self->addClassProperty( fclass, "NORMAL" )
      .setInteger(HPDF_BM_NORMAL).setReadOnly(true);
    self->addClassProperty( fclass, "MULTIPLY" )
      .setInteger(HPDF_BM_MULTIPLY).setReadOnly(true);
    self->addClassProperty( fclass, "SCREEN" )
      .setInteger(HPDF_BM_SCREEN).setReadOnly(true);
    self->addClassProperty( fclass, "OVERLAY" )
      .setInteger(HPDF_BM_OVERLAY).setReadOnly(true);
    self->addClassProperty( fclass, "DARKEN" )
      .setInteger(HPDF_BM_DARKEN).setReadOnly(true);
    self->addClassProperty( fclass, "LIGHTEN" )
      .setInteger(HPDF_BM_LIGHTEN).setReadOnly(true);
    self->addClassProperty( fclass, "COLOR_DODGE" )
      .setInteger(HPDF_BM_COLOR_DODGE).setReadOnly(true);
    self->addClassProperty( fclass, "COLOR_BUM" )
      .setInteger(HPDF_BM_COLOR_BUM).setReadOnly(true);
    self->addClassProperty( fclass, "HARD_LIGHT" )
      .setInteger(HPDF_BM_HARD_LIGHT).setReadOnly(true);
    self->addClassProperty( fclass, "SOFT_LIGHT" )
      .setInteger(HPDF_BM_SOFT_LIGHT).setReadOnly(true);
    self->addClassProperty( fclass, "DIFFERENCE" )
      .setInteger(HPDF_BM_DIFFERENCE).setReadOnly(true);
    self->addClassProperty( fclass, "EXCLUSHON" )
      .setInteger(HPDF_BM_EXCLUSHON).setReadOnly(true);
  }

  {
    Falcon::Symbol* fclass = self->addClass( "TransitionStyle" );
    fclass->setEnum(true);
    self->addClassProperty( fclass, "WIPE_RIGHT" )
      .setInteger(HPDF_TS_WIPE_RIGHT).setReadOnly(true);
    self->addClassProperty( fclass, "WIPE_UP" )
      .setInteger(HPDF_TS_WIPE_UP).setReadOnly(true);
    self->addClassProperty( fclass, "WIPE_LEFT" )
      .setInteger(HPDF_TS_WIPE_LEFT).setReadOnly(true);
    self->addClassProperty( fclass, "WIPE_DOWN" )
      .setInteger(HPDF_TS_WIPE_DOWN).setReadOnly(true);
    self->addClassProperty( fclass, "BARN_DOORS_HORIZONTAL_OUT" )
      .setInteger(HPDF_TS_BARN_DOORS_HORIZONTAL_OUT).setReadOnly(true);
    self->addClassProperty( fclass, "BARN_DOORS_HORIZONTAL_IN" )
      .setInteger(HPDF_TS_BARN_DOORS_HORIZONTAL_IN).setReadOnly(true);
    self->addClassProperty( fclass, "BARN_DOORS_VERTICAL_OUT" )
      .setInteger(HPDF_TS_BARN_DOORS_VERTICAL_OUT).setReadOnly(true);
    self->addClassProperty( fclass, "BARN_DOORS_VERTICAL_IN" )
      .setInteger(HPDF_TS_BARN_DOORS_VERTICAL_IN).setReadOnly(true);
    self->addClassProperty( fclass, "BOX_OUT" )
      .setInteger(HPDF_TS_BOX_OUT).setReadOnly(true);
    self->addClassProperty( fclass, "BOX_IN" )
      .setInteger(HPDF_TS_BOX_IN).setReadOnly(true);
    self->addClassProperty( fclass, "BLINDS_HORIZONTAL" )
      .setInteger(HPDF_TS_BLINDS_HORIZONTAL).setReadOnly(true);
    self->addClassProperty( fclass, "BLINDS_VERTICAL" )
      .setInteger(HPDF_TS_BLINDS_VERTICAL).setReadOnly(true);
    self->addClassProperty( fclass, "DISSOLVE" )
      .setInteger(HPDF_TS_DISSOLVE).setReadOnly(true);
    self->addClassProperty( fclass, "GLITTER_RIGHT" )
      .setInteger(HPDF_TS_GLITTER_RIGHT).setReadOnly(true);
    self->addClassProperty( fclass, "GLITTER_DOWN" )
      .setInteger(HPDF_TS_GLITTER_DOWN).setReadOnly(true);
    self->addClassProperty( fclass, "GLITTER_TOP_LEFT_TO_BOTTOM_RIGHT" )
      .setInteger(HPDF_TS_GLITTER_TOP_LEFT_TO_BOTTOM_RIGHT).setReadOnly(true);
    self->addClassProperty( fclass, "REPLACE" )
      .setInteger(HPDF_TS_REPLACE).setReadOnly(true);
  }

  {
    Falcon::Symbol* fclass = self->addClass( "PageSize" );
    fclass->setEnum(true);
    self->addClassProperty( fclass, "LETTER" )
      .setInteger(HPDF_PAGE_SIZE_LETTER).setReadOnly(true);
    self->addClassProperty( fclass, "LEGAL" )
      .setInteger(HPDF_PAGE_SIZE_LEGAL).setReadOnly(true);
    self->addClassProperty( fclass, "A3" )
      .setInteger(HPDF_PAGE_SIZE_A3).setReadOnly(true);
    self->addClassProperty( fclass, "A4" )
      .setInteger(HPDF_PAGE_SIZE_A4).setReadOnly(true);
    self->addClassProperty( fclass, "A5" )
      .setInteger(HPDF_PAGE_SIZE_A5).setReadOnly(true);
    self->addClassProperty( fclass, "B4" )
      .setInteger(HPDF_PAGE_SIZE_B4).setReadOnly(true);
    self->addClassProperty( fclass, "B5" )
      .setInteger(HPDF_PAGE_SIZE_B5).setReadOnly(true);
    self->addClassProperty( fclass, "EXECUTIVE" )
      .setInteger(HPDF_PAGE_SIZE_EXECUTIVE).setReadOnly(true);
    self->addClassProperty( fclass, "US4x6" )
      .setInteger(HPDF_PAGE_SIZE_US4x6).setReadOnly(true);
    self->addClassProperty( fclass, "US4x8" )
      .setInteger(HPDF_PAGE_SIZE_US4x8).setReadOnly(true);
    self->addClassProperty( fclass, "US5x7" )
      .setInteger(HPDF_PAGE_SIZE_US5x7).setReadOnly(true);
    self->addClassProperty( fclass, "COMM10" )
      .setInteger(HPDF_PAGE_SIZE_COMM10).setReadOnly(true);
  }

  {
    Falcon::Symbol* fclass = self->addClass( "PageDirection" );
    fclass->setEnum(true);
    self->addClassProperty( fclass, "PORTRAIT" )
      .setInteger(HPDF_PAGE_PORTRAIT).setReadOnly(true);
    self->addClassProperty( fclass, "LANDSCAPE" )
      .setInteger(HPDF_PAGE_LANDSCAPE).setReadOnly(true);
  }

  {
    Falcon::Symbol* fclass = self->addClass( "EncoderType" );
    fclass->setEnum(true);
    self->addClassProperty( fclass, "SINGLE_BYTE" )
      .setInteger(HPDF_ENCODER_TYPE_SINGLE_BYTE).setReadOnly(true);
    self->addClassProperty( fclass, "DOUBLE_BYTE" )
      .setInteger(HPDF_ENCODER_TYPE_DOUBLE_BYTE).setReadOnly(true);
    self->addClassProperty( fclass, "UNINITIALIZED" )
      .setInteger(HPDF_ENCODER_TYPE_UNINITIALIZED).setReadOnly(true);
    self->addClassProperty( fclass, "UNKNOWN" )
      .setInteger(HPDF_ENCODER_UNKNOWN).setReadOnly(true);
  }

  {
    Falcon::Symbol* fclass = self->addClass( "ByteType" );
    fclass->setEnum(true);
    self->addClassProperty( fclass, "SINGLE" )
      .setInteger(HPDF_BYTE_TYPE_SINGLE).setReadOnly(true);
    self->addClassProperty( fclass, "LEAD" )
      .setInteger(HPDF_BYTE_TYPE_LEAD).setReadOnly(true);
    self->addClassProperty( fclass, "TRIAL" )
      .setInteger(HPDF_BYTE_TYPE_TRIAL).setReadOnly(true);
    self->addClassProperty( fclass, "UNKNOWN" )
      .setInteger(HPDF_BYTE_TYPE_UNKNOWN).setReadOnly(true);
  }

  {
    /*#
       @enum TextAlignment
       @brief Specifies the text alignment to be used.

       Possible valuse:
        - LEFT
        - RIGHT
        - CENTER
        - JUSTIFY
     */
    Falcon::Symbol* fclass = self->addClass( "TextAlignment" );
    fclass->setEnum(true);
    self->addClassProperty( fclass, "LEFT" )
      .setInteger(HPDF_TALIGN_LEFT).setReadOnly(true);
    self->addClassProperty( fclass, "RIGHT" )
      .setInteger(HPDF_TALIGN_RIGHT).setReadOnly(true);
    self->addClassProperty( fclass, "CENTER" )
      .setInteger(HPDF_TALIGN_CENTER).setReadOnly(true);
    self->addClassProperty( fclass, "JUSTIFY" )
      .setInteger(HPDF_TALIGN_JUSTIFY).setReadOnly(true);
  }
}
}}} // Falcon::Ext::hpdf
