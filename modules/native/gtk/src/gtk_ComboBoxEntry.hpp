#ifndef GTK_COMBOBOXENTRY_HPP
#define GTK_COMBOBOXENTRY_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::ComboBoxEntry
 */
class ComboBoxEntry
    :
    public Gtk::CoreGObject
{
public:

    ComboBoxEntry( const Falcon::CoreClass*, const GtkComboBoxEntry* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    //static FALCON_FUNC new_with_model( VMARG );

    static FALCON_FUNC new_text( VMARG );

    static FALCON_FUNC set_text_column( VMARG );

    static FALCON_FUNC get_text_column( VMARG );

};

} // Gtk
} // Falcon

#endif // !GTK_COMBOBOXENTRY_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
