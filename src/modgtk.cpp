/**
 *  \file modgtk.cpp
 */

#include "modgtk.hpp"

#include "g_Object.hpp"
#include "g_ParamSpec.hpp"

#include "gdk_DragContext.hpp"
#include "gdk_EventButton.hpp"
#include "gdk_Pixbuf.hpp"

#include "gtk_enums.hpp"

#include "gtk_Action.hpp"
#include "gtk_Adjustment.hpp"
#include "gtk_Alignment.hpp"
#include "gtk_Arrow.hpp"
#include "gtk_AspectFrame.hpp"
#include "gtk_Bin.hpp"
#include "gtk_Box.hpp"
#include "gtk_Button.hpp"
#include "gtk_ButtonBox.hpp"
#include "gtk_CheckButton.hpp"
#include "gtk_ComboBox.hpp"
#include "gtk_ComboBoxEntry.hpp"
#include "gtk_Container.hpp"
#include "gtk_Entry.hpp"
#include "gtk_EntryBuffer.hpp"
#include "gtk_EventBox.hpp"
#include "gtk_Expander.hpp"
#include "gtk_Fixed.hpp"
#include "gtk_Frame.hpp"
#include "gtk_HBox.hpp"
#include "gtk_HButtonBox.hpp"
#include "gtk_HPaned.hpp"
#include "gtk_Image.hpp"
#include "gtk_Label.hpp"
#include "gtk_Layout.hpp"
#include "gtk_Main.hpp"
#include "gtk_Misc.hpp"
#include "gtk_Object.hpp"
#include "gtk_Paned.hpp"
#include "gtk_RadioAction.hpp"
#include "gtk_RadioButton.hpp"
#include "gtk_Requisition.hpp"
#include "gtk_Stock.hpp"
#include "gtk_Table.hpp"
#include "gtk_TextTag.hpp"
#include "gtk_TextTagTable.hpp"
#include "gtk_ToggleAction.hpp"
#include "gtk_ToggleButton.hpp"
#include "gtk_VBox.hpp"
#include "gtk_VButtonBox.hpp"
#include "gtk_VPaned.hpp"
#include "gtk_Widget.hpp"
#include "gtk_Window.hpp"


FALCON_MODULE_DECL
{
    #define FALCON_DECLARE_MODULE self

    Falcon::Module* self = new Falcon::Module();
    self->name( "gtk" );
    self->language( "en_US" );
    self->engineVersion( FALCON_VERSION_NUM );
    self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

    #include "modgtk_st.hpp"

    /*
     *  load glib
     */

    Falcon::Glib::Object::modInit( self );
    Falcon::Glib::ParamSpec::modInit( self );

    /*
     *  load gdk
     */

    Falcon::Gdk::DragContext::modInit( self );
    Falcon::Gdk::EventButton::modInit( self );
    Falcon::Gdk::Pixbuf::modInit( self );

    /*
     *  load enums
     */
    Falcon::Gtk::Enums::modInit( self );

    /*
     *  setup the classes
     */

    Falcon::Gtk::Main::modInit( self );

    Falcon::Gtk::Requisition::modInit( self );
    Falcon::Gtk::Stock::modInit( self );

    Falcon::Gtk::Action::modInit( self );
        Falcon::Gtk::ToggleAction::modInit( self );
            Falcon::Gtk::RadioAction::modInit( self );
#if GTK_VERSION_MINOR >= 18
    Falcon::Gtk::EntryBuffer::modInit( self );
#endif
    Falcon::Gtk::Object::modInit( self );
        Falcon::Gtk::Adjustment::modInit( self );
        Falcon::Gtk::Widget::modInit( self );
            Falcon::Gtk::Container::modInit( self );
                Falcon::Gtk::Bin::modInit( self );
                    Falcon::Gtk::Alignment::modInit( self );
                    Falcon::Gtk::Button::modInit( self );
                        Falcon::Gtk::ToggleButton::modInit( self );
                            Falcon::Gtk::CheckButton::modInit( self );
                                Falcon::Gtk::RadioButton::modInit( self );
                    Falcon::Gtk::ComboBox::modInit( self );
                        Falcon::Gtk::ComboBoxEntry::modInit( self );
                    Falcon::Gtk::EventBox::modInit( self );
                    Falcon::Gtk::Expander::modInit( self );
                    Falcon::Gtk::Frame::modInit( self );
                        Falcon::Gtk::AspectFrame::modInit( self );
                    Falcon::Gtk::Window::modInit( self );
                Falcon::Gtk::Box::modInit( self );
                    Falcon::Gtk::ButtonBox::modInit( self );
                        Falcon::Gtk::HButtonBox::modInit( self );
                        Falcon::Gtk::VButtonBox::modInit( self );
                    Falcon::Gtk::HBox::modInit( self );
                    Falcon::Gtk::VBox::modInit( self );
                Falcon::Gtk::Fixed::modInit( self );
                Falcon::Gtk::Layout::modInit( self );
                Falcon::Gtk::Paned::modInit( self );
                    Falcon::Gtk::HPaned::modInit( self );
                    Falcon::Gtk::VPaned::modInit( self );
                Falcon::Gtk::Table::modInit( self );
            Falcon::Gtk::Entry::modInit( self );
            Falcon::Gtk::Misc::modInit( self );
                Falcon::Gtk::Arrow::modInit( self );
                Falcon::Gtk::Image::modInit( self );
                Falcon::Gtk::Label::modInit( self );
    Falcon::Gtk::TextTag::modInit( self );
    Falcon::Gtk::TextTagTable::modInit( self );

    return self;
}


namespace Falcon {
namespace Gtk {


GObject* internal_add_slot( GObject* obj )
{
    Falcon::CoreSlot* cs = new Falcon::CoreSlot( "" );

    g_object_set_data_full( obj, "_signals", (gpointer) cs,
        &Gtk::internal_release_slot );

    return obj;
}


void internal_release_slot( gpointer cs )
{
    ((Falcon::CoreSlot*)cs)->clear();
    ((Falcon::CoreSlot*)cs)->decref();
}


void internal_get_slot( const char* signame, void* cb, Falcon::VMachine* vm )
{
    MYSELF;
    GET_OBJ( self );
    GET_SIGNALS( _obj );
    CoreSlot* ev = get_signal( _obj, _signals, signame, cb, vm );
    vm->retval( vm->findWKI( "VMSlot" )->asClass()->createInstance( ev ) );
}


void internal_trigger_slot( GObject* obj, const char* signame,
        const char* cbname, Falcon::VMachine* vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( signame, false );

    if ( !cs || cs->empty() )
        return;

    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();
        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( cbname, it ) )
            {
                printf( "[%s] invalid callback (expected callable)\n", cbname );
                return;
            }
        }
        vm->callItem( it, 0 );
        iter.next();
    }
    while ( iter.hasCurrent() );
}


FALCON_FUNC abstract_init( VMARG )
{
    MYSELF;

    if ( !self->getUserData() )
    {
        throw_gtk_error( e_abstract_class, FAL_STR( gtk_e_abstract_class_ ) );
    }
}


Falcon::CoreSlot* get_signal( GObject* obj, Falcon::CoreSlot* sl,
    const char* name, void* cb, Falcon::VMachine* vm )
{
    Falcon::CoreSlot* cs = sl->getChild( name, false );
    if ( !cs )
    {
        cs = sl->getChild( name, true );
        g_signal_connect( G_OBJECT( obj ), name,
                          G_CALLBACK( cb ), vm );
    }
    return cs;
}


bool CoreGObject::getProperty( const Falcon::String& s, Falcon::Item& it ) const
{
    GObject* obj = ((GData*)getUserData())->obj();
    AutoCString cstr( s );
    Item* itm = (Item*) g_object_get_data( obj, cstr.c_str() );
    if ( itm )
        it = *itm;
    else
        return defaultProperty( s, it );
    return true;
}


bool CoreGObject::setProperty( const Falcon::String& s, const Falcon::Item& it )
{
    GObject* obj = ((GData*)getUserData())->obj();
    AutoCString cstr( s );
    g_object_set_data_full( obj, cstr.c_str(), new Item( it ), &CoreGObject::delProperty );
    return true;
}


void CoreGObject::delProperty( gpointer it )
{
    delete (Item*) it;
}


/*#
    @class GtkError
    @brief Error generated by falcon-gtk
    @optparam code numeric error code
    @optparam description textual description of the error code
    @optparam extra descriptive message explaining the error conditions
    @from Error code, description, extra

    See the Error class in the core module.
*/
FALCON_FUNC GtkError_init( VMARG )
{
    MYSELF;

    if ( !self->getUserData() )
        self->setUserData( new Falcon::Gtk::GtkError );

    Falcon::core::Error_init( vm );
}


} // Gtk
} // Falcon
