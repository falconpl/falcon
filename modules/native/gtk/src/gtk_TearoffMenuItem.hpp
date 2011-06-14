#ifndef GTK_TEAROFFMENUITEM_HPP
#define GTK_TEAROFFMENUITEM_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::TearoffMenuItem
 */
class TearoffMenuItem
    :
    public Gtk::CoreGObject
{
public:

    TearoffMenuItem( const Falcon::CoreClass*, const GtkTearoffMenuItem* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_TEAROFFMENUITEM_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
