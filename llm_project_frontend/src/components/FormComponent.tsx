"use client"

import { zodResolver } from "@hookform/resolvers/zod"
import { useForm } from "react-hook-form"
import { z } from "zod"

import { Button } from "@/components/ui/button"
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form"
import { Input } from "@/components/ui/input"
import { useState } from "react"

const formSchema = z.object({
  username: z.string().min(2, {
    message: "Username must be at least 2 characters.",
  }).max(20, {
    message: "Username must be at most 20 characters.",
  }),
})

export function FormComponent() {
  const [responseData, setResponseData] = useState(null);
  const [loading, setLoading] = useState(false);

  // 1. Define your form.
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      username: "",
    },
  })

  // 2. Define a submit handler.
  const onSubmit = async (e: z.infer<typeof formSchema>) => {
    console.log("e: ",e);
    
    setLoading(true);
    setResponseData(null);
    try {
      const response = await fetch(`http://localhost:8000/query/${e.username}`);
      console.log("response: ",response);
      
      const data = await response.json();
      setResponseData(data.reponse);
      console.log("data: ",data.response);
      
    } catch (error) {
      console.error(error)
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
          <FormField
            control={form.control}
            name="username"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Username</FormLabel>
                <FormControl>
                  <Input placeholder="shadcn" {...field} />
                </FormControl>
                <FormDescription>
                  This is your public display name.
                </FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />
          <Button type="submit"> {loading ? "Enviando..." : "Enviar"}</Button>
        </form>
      </Form>
      {responseData && (
        <div className="mt-6 p-4 bg-zinc-700 rounded-lg shadow-md w-full max-w-md text-gray-100">
          <p className="text-lg font-semibold">Respuesta:</p>
          <p>{responseData}</p>
        </div>
      )}
    </>

  )
}
