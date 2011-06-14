#ifndef GTK_CHECKBUTTON_HPP
#define GTK_CHECKBUTTON_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::CheckButton
 */
class CheckButton
    :
    public Gtk::CoreGObject
{
public:

    CheckButton( const Falcon::CoreClass*, const GtkCheckButton* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC new_with_label( VMARG );

    static FALCON_FUNC new_with_mnemonic( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_CHECKBUTTON_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
