/*
   Mini XML lib PLUS for C++

   Main definitions

   Author: Giancarlo Niccolai <gian@niccolai.ws>

*/

/** \file
   Standard definitions for MXML.

   This files contains MXML macros used commonly for bitfield oriented
   style definitions and prepares the environment.

   This also includes mxml_document.h, which includes in turn all the
   necessary files.

   It should be included by all files using MXML.
*/
#ifndef MXML_H
#define MXML_H

#define MXML_LINE_TERMINATOR      '\n'
#define MXML_SOFT_LINE_TERMINATOR '\r'
#define MXML_PATH_SEPARATOR       '/'

/** @group stylemacros Bitfield oriented macros defining MXML output style.

   MXML is able to create XML document dynamically, if it is provided with a
   set of node; this bitfield macros can be used to specify
   to have more nicely formatted output, so that the file is also well readable
   by a human reader. If the file must be sent to a stream and it is not meant
   to be used by humans, it's quite useless to format it as it's just a waste
   of time on both sides.

   Usually, you will want to format correctly files that are going to be saved
   on the hard disk and that may be inspected by users and eventually changed
   with human intervention.

   Some of the styles affects also how the document is loaded (parsed),
   i.e. preveinting automatic translation of escaped characters.

   Some fields can be combined with the others by ORing them up.
*/

/** Indent correctly the document.

   @addtogroup stylemacros
   If no other bitfield is specificed, only one blank (usually ' ' == 0x20) is
   used as indentation; MXML_STYLE_TAB and MXML_STYLE_THREESPACES may be used
   in conjunction with this field.
*/
#define MXML_STYLE_INDENT        0x0001

/** Indent using tabs (0x08) instead of spaces
   @addtogroup stylemacros
*/
#define MXML_STYLE_TAB           0x0002

/** Each level is indented using three indentation level instead of one.
   @addtogroup stylemacros

   Must be used in conjunction with MXML_STYLE_INDENT.
*/
#define MXML_STYLE_THREESPACES   0x0004

/** Avoid escaping XML entities as &quot; and &amper;, or unescaping them while reading.
   @addtogroup stylemacros

   You may specify it before reading the XML to avoid having the escape sequence
   automatically translated on loading; also, this will cause the output to be put on
   disk as untraslated.

   If you know your file has no escape characters, or you don't need them to be
   escaped, this option saves you a considerable time.
*/
#define MXML_STYLE_NOESCAPE      0x0008

#include "mxml_document.h"
#include "mxml_error.h"

#endif
