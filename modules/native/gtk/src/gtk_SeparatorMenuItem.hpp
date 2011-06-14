#ifndef GTK_SEPARATORMENUITEM_HPP
#define GTK_SEPARATORMENUITEM_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::SeparatorMenuItem
 */
class SeparatorMenuItem
    :
    public Gtk::CoreGObject
{
public:

    SeparatorMenuItem( const Falcon::CoreClass*, const GtkSeparatorMenuItem* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_SEPARATORMENUITEM_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
