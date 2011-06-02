#ifndef GTK_HBUTTONBOX_HPP
#define GTK_HBUTTONBOX_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::HButtonBox
 */
class HButtonBox
    :
    public Gtk::CoreGObject
{
public:

    HButtonBox( const Falcon::CoreClass*, const GtkHButtonBox* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    //static FALCON_FUNC get_spacing_default( VMARG );

    //static FALCON_FUNC get_layout_default( VMARG );

    //static FALCON_FUNC set_spacing_default( VMARG );

    //static FALCON_FUNC set_layout_default( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_HBUTTONBOX_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
