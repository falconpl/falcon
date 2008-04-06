/*
   FALCON - The Falcon Programming Language.
   FILE: labeldef.h

   Definition for assembly oriented labels.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab ago 27 2005

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Definition for assembly oriented labels.
*/

#ifndef flc_labeldef_H
#define flc_labeldef_H

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/types.h>
#include <falcon/genericlist.h>
#include <falcon/basealloc.h>

namespace Falcon {

#define FASM_UNDEFINED_LABEL 0xFFFFFFFF

class String;
class Stream;

/** Records informations about the label definitions in an assembly file.

   In assembly files, labels can be referenced before their declaration and
   declared in any moment. However, the module being compiled is necessarily
   built both if the label target address is known or unknown.

   Moreover, label names are string that must be disposed of when the
   module compilation is complete.

   This class is used to keep track of all the information needed to assembly
   the labels during compilation, and to discard their names as soon as the
   compilation is complete.

   The class also takes care to fill in the real position of the label
   into the dummy area in the module stream when the label is defined. If
   there were some previous reference to the label that is now being defined,
   then the stream is searched for the previous references and the current
   value of the label is filled in.

*/
class FALCON_DYN_CLASS LabelDef: public BaseAlloc
{

private:

   String m_name;
   uint32 m_position;
   List m_forwards;
   void addForwardRef( uint32 pos ) { m_forwards.pushBack( (void *) pos ); }

public:

   /** Creates the labeldef.

      The string passes as parameter is owned by this class and
      destroyed at object termination.

   \param name the name of this label.
   */
   LabelDef( const String &name );

   /** Destroys the label definition.
      The String that holds internally the name of this label is
      destroyed too.
   */
   ~LabelDef();

   /** Write the label on the file.
      If the position of the label definition has already been determined,
      the value of the label will be written on the stream, being a valid
      identificator for the label in the VM code. If it's not already defined,
      then a dummy value will be written and the position will be recorded
      as a forward reference.

      As soon as the real position of the label is found and the defineNow()
      method is called, all the existing forward references are overwritten
      with the actual value of the label.
      \param os the ouptut stream where to write the label on.
   */
   void write( Stream *os );

   /** Determines wether the label has been defined or not.
      \return true if the label has been defined, false otherwise.
   */
   bool defined() const { return ( m_position != FASM_UNDEFINED_LABEL); }

   /** Define the label at current position in the stream.
      The label is defined and the position where it can be found
      is set at the current write position of the stream. Also,
      if some forward references were previously made, the dummy
      values saved on the file will be overwritten (provided they
      were recorded with addForwardRef().
   */
   void defineNow( Stream *os );
   const String &name() const { return m_name; }
};

}

#endif

/* end of labeldef.h */
