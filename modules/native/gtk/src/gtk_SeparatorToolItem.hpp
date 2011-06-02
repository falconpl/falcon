#ifndef GTK_SEPARATORTOOLITEM_HPP
#define GTK_SEPARATORTOOLITEM_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::SeparatorToolItem
 */
class SeparatorToolItem
    :
    public Gtk::CoreGObject
{
public:

    SeparatorToolItem( const Falcon::CoreClass*, const GtkSeparatorToolItem* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC set_draw( VMARG );

    static FALCON_FUNC get_draw( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_SEPARATORTOOLITEM_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
